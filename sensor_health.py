"""Read-only, position-aware Sensor Health / Timeline Diagnostics model.

Answers "what is available at the current playback position?" -- as
opposed to validator.py's DatasetIntegrityValidator, which answers "is the
recording structurally usable?" at the session level. This module never
reads a file, never decodes anything, and never recomputes a nearest-
timestamp search: it only reads values already sitting in memory --
RecordingSession.frames/radar/events/mti/lidar, an already-open SvoReader,
the already-built SyncIndex, and DatasetIntegrityValidator's already-
computed ValidationResult -- and reports them.

Nothing here changes which sample gets selected, alters a timestamp, fuses
sensors, or infers a dropped sample from a sample-count mismatch. A sensor
is only ever reported DROPPED when a recorded event's own timestamp exactly
matches the recorded sample's own timestamp -- i.e. only when the dataset's
own metadata says so, never as an inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from sync_index import MISSING

# Display-only defaults (see SyncThresholds below). Changing these only
# changes how an already-selected delta is labeled; it can never change
# which sample SyncIndex selected, nor any recorded sensor value.
DEFAULT_SYNC_GOOD_THRESHOLD_MS = 20.0
DEFAULT_SYNC_WARNING_THRESHOLD_MS = 60.0


class SensorState(str, Enum):
    AVAILABLE = "AVAILABLE"     # a recorded sample is selected and within the warning threshold
    NO_DATA = "NO_DATA"         # this sensor has no data at all for this session/frame
    NO_FIX = "NO_FIX"           # GNSS-specific: MTi loaded, but no valid lat/lon fix recorded
    STALE = "STALE"             # a sample is selected, but |delta t| exceeds the warning threshold
    DROPPED = "DROPPED"         # a recorded drop event's own timestamp matches this exact sample
    INVALID = "INVALID"         # the recorded timestamp itself is non-finite
    UNAVAILABLE = "UNAVAILABLE"  # an optional dependency (e.g. ouster-sdk) is not installed

    def display(self) -> str:
        return self.value.replace("_", " ")


@dataclass
class SyncThresholds:
    """Display-only thresholds for classifying an already-computed |delta t|.

    These never influence which sample SyncIndex selects, never suppress or
    alter data, and never touch playback -- they only decide which of
    GOOD/WARNING/STALE label gets attached to a number that already exists.
    """

    good_ms: float = DEFAULT_SYNC_GOOD_THRESHOLD_MS
    warning_ms: float = DEFAULT_SYNC_WARNING_THRESHOLD_MS

    def classify(self, abs_delta_ms: float) -> str:
        if abs_delta_ms <= self.good_ms:
            return "GOOD"
        if abs_delta_ms <= self.warning_ms:
            return "WARNING"
        return "STALE"


@dataclass
class SensorDiagnostic:
    name: str
    state: SensorState
    sample_index: Optional[int] = None
    sensor_timestamp_ns: Optional[int] = None
    delta_ns: Optional[int] = None  # sensor_timestamp - camera_timestamp, signed
    sync_class: Optional[str] = None  # "GOOD"/"WARNING"/"STALE", display-only
    detail: str = ""

    @property
    def delta_ms(self) -> Optional[float]:
        return None if self.delta_ns is None else self.delta_ns / 1e6


@dataclass
class TimelineEvent:
    event_type: str
    description: str
    timestamp_ns: int


@dataclass
class SensorHealthSnapshot:
    camera_timestamp_ns: int
    camera_frame_index: int
    camera: SensorDiagnostic
    radar: SensorDiagnostic
    lidar: SensorDiagnostic
    mti: SensorDiagnostic
    zed_imu: SensorDiagnostic
    gnss: SensorDiagnostic

    def strip_rows(self):
        """The six rows the compact status strip shows."""
        return [
            ("Camera", self.camera.state),
            ("Radar", self.radar.state),
            ("Lidar", self.lidar.state),
            ("MTi-8", self.mti.state),
            ("ZED IMU", self.zed_imu.state),
            ("GNSS", self.gnss.state),
        ]

    def detail_rows(self):
        return [
            ("RADAR", self.radar),
            ("LIDAR", self.lidar),
            ("MTI-8", self.mti),
            ("ZED IMU", self.zed_imu),
            ("GNSS", self.gnss),
        ]


def _has_valid_fix(latitude: np.ndarray, longitude: np.ndarray) -> bool:
    """Identical check to validator._has_valid_fix -- kept as a fallback for
    when a ValidationResult isn't available; used against already-loaded
    MTi arrays, never re-read from disk."""
    return bool(np.any(np.isfinite(latitude) & np.isfinite(longitude) & (latitude != 0) & (longitude != 0)))


class SensorHealthModel:
    """Builds one SensorHealthSnapshot per requested camera frame index.

    Everything it reads was already loaded before this model is asked for
    anything: RecordingSession's dataframes/MtiData/LidarReader, the
    already-open SvoReader, the already-built SyncIndex, and (optionally)
    DatasetIntegrityValidator's already-computed ValidationResult. Building
    a snapshot never touches disk and never blocks -- it's a handful of
    array indexing operations, safe to call synchronously on every seek.
    """

    def __init__(self, session, sync_index, svo=None, validation_result=None, thresholds: SyncThresholds | None = None):
        self.session = session
        self.sync_index = sync_index
        self.svo = svo
        self.validation_result = validation_result
        self.thresholds = thresholds or SyncThresholds()
        self._drop_events = self._collect_drop_events()

    def _collect_drop_events(self) -> list:
        events = self.session.events
        if events is None or events.empty:
            return []
        # Same substring rule DatasetIntegrityValidator already uses to spot
        # drop/counter events -- reused here rather than reinvented.
        mask = events["event_type"].astype(str).str.contains("DROP", case=False, na=False)
        rows = events[mask]
        return [
            TimelineEvent(
                event_type=str(r["event_type"]),
                description=str(r["description"]),
                timestamp_ns=int(r["timestamp_ns"]),
            )
            for _, r in rows.iterrows()
        ]

    @property
    def drop_events(self) -> list:
        return list(self._drop_events)

    def snapshot(self, frame_index: int) -> SensorHealthSnapshot:
        frames = self.session.frames
        row = frames.iloc[frame_index]
        camera_ts = int(row["timestamp_ns"])

        camera_diag = SensorDiagnostic(
            name="Camera", state=SensorState.AVAILABLE,
            sample_index=frame_index, sensor_timestamp_ns=camera_ts, delta_ns=0, sync_class="GOOD",
        )

        return SensorHealthSnapshot(
            camera_timestamp_ns=camera_ts,
            camera_frame_index=frame_index,
            camera=camera_diag,
            radar=self._radar_diagnostic(frame_index, camera_ts),
            lidar=self._lidar_diagnostic(frame_index, camera_ts),
            mti=self._mti_diagnostic(frame_index, camera_ts),
            zed_imu=self._zed_imu_diagnostic(frame_index, camera_ts),
            gnss=self._gnss_diagnostic(),
        )

    # ------------------------------------------------------------ helpers ---
    def _finalize(self, name: str, sample_index, sensor_ts: int, delta_ns: int, drop_event_types: frozenset) -> SensorDiagnostic:
        if not np.isfinite(sensor_ts):
            return SensorDiagnostic(name, SensorState.INVALID, sample_index=sample_index,
                                     detail="non-finite recorded timestamp")

        abs_ms = abs(delta_ns) / 1e6
        sync_class = self.thresholds.classify(abs_ms)

        # DROPPED only when a recorded event's own timestamp is the exact
        # timestamp of this exact recorded sample -- evidence from the
        # dataset's own metadata, never an inference from a sample-count
        # mismatch or from proximity to the current camera frame.
        is_dropped = any(
            ev.event_type in drop_event_types and ev.timestamp_ns == sensor_ts
            for ev in self._drop_events
        )

        if is_dropped:
            state = SensorState.DROPPED
        elif sync_class == "STALE":
            state = SensorState.STALE
        else:
            state = SensorState.AVAILABLE

        return SensorDiagnostic(
            name, state, sample_index=sample_index, sensor_timestamp_ns=sensor_ts,
            delta_ns=delta_ns, sync_class=sync_class,
        )

    def _radar_diagnostic(self, frame_index: int, camera_ts: int) -> SensorDiagnostic:
        radar_df = self.session.radar
        if radar_df is None or radar_df.empty:
            return SensorDiagnostic("Radar", SensorState.NO_DATA, detail="no radar metadata for this session")
        pos = self.sync_index.radar_for(frame_index)
        if pos == MISSING:
            return SensorDiagnostic("Radar", SensorState.NO_DATA, detail="no synchronized sample")
        row = radar_df.iloc[pos]
        sensor_ts = int(row["timestamp_ns"])
        sample_index = int(row["frame_id"]) if "frame_id" in row else int(pos)
        return self._finalize("Radar", sample_index, sensor_ts, sensor_ts - camera_ts,
                               frozenset({"RADAR_DROPPED_FRAMES"}))

    def _lidar_diagnostic(self, frame_index: int, camera_ts: int) -> SensorDiagnostic:
        lidar = self.session.lidar
        if lidar is None:
            if self.validation_result is not None:
                unavailable = any(c.status.value == "UNAVAILABLE" for c in self.validation_result.lidar.checks)
                if unavailable:
                    return SensorDiagnostic("Lidar", SensorState.UNAVAILABLE, detail="lidar dependency not installed")
            return SensorDiagnostic("Lidar", SensorState.NO_DATA, detail="no lidar data for this session")
        if lidar.num_scans == 0:
            return SensorDiagnostic("Lidar", SensorState.NO_DATA, detail="0 scans")
        idx = self.sync_index.lidar_for(frame_index)
        if idx == MISSING:
            return SensorDiagnostic("Lidar", SensorState.NO_DATA, detail="no synchronized sample")
        sensor_ts = int(lidar.timestamps_ns[idx])
        return self._finalize("Lidar", idx, sensor_ts, sensor_ts - camera_ts,
                               frozenset({"LIDAR_DROPPED_FRAMES"}))

    def _mti_diagnostic(self, frame_index: int, camera_ts: int) -> SensorDiagnostic:
        mti = self.session.mti
        if mti is None or mti.n == 0:
            return SensorDiagnostic("MTi-8", SensorState.NO_DATA, detail="no MTi data for this session")
        idx = self.sync_index.mti_for(frame_index)
        if idx == MISSING:
            return SensorDiagnostic("MTi-8", SensorState.NO_DATA, detail="no synchronized sample")
        sensor_ts = int(mti.timestamp_ns[idx])
        return self._finalize("MTi-8", idx, sensor_ts, sensor_ts - camera_ts, frozenset())

    def _zed_imu_diagnostic(self, frame_index: int, camera_ts: int) -> SensorDiagnostic:
        svo = self.svo
        if svo is None:
            return SensorDiagnostic("ZED IMU", SensorState.NO_DATA, detail="camera not opened")
        if not getattr(svo, "has_zed_imu", False):
            return SensorDiagnostic("ZED IMU", SensorState.NO_DATA, detail="not recorded in this SVO2")
        idx = self.sync_index.svo_for(frame_index)
        if idx == MISSING:
            return SensorDiagnostic("ZED IMU", SensorState.NO_DATA, detail="no synchronized sample")
        sensor_ts = int(svo.timestamps_ns[idx])
        return self._finalize("ZED IMU", idx, sensor_ts, sensor_ts - camera_ts, frozenset())

    def _gnss_diagnostic(self) -> SensorDiagnostic:
        """Preserves the exact GNSS semantics DatasetIntegrityValidator
        already implements (raw NMEA is not the deciding factor; MTi fused
        position is) -- reusing its already-computed flag when available
        rather than recomputing it."""
        mti = self.session.mti
        if mti is None or mti.n == 0:
            return SensorDiagnostic("GNSS", SensorState.NO_DATA, detail="no MTi data (GNSS is derived from MTi fusion)")

        has_fix = None
        if self.validation_result is not None:
            has_fix = self.validation_result.mti.info.get("has_valid_fix")
        if has_fix is None:
            has_fix = _has_valid_fix(mti.latitude, mti.longitude)

        if has_fix:
            return SensorDiagnostic("GNSS", SensorState.AVAILABLE, detail="available via MTi fused position")
        return SensorDiagnostic("GNSS", SensorState.NO_FIX, detail="no valid fix in this recording")
