"""Read-only Dataset Integrity Validator.

Inspects an already-loaded RecordingSession (and, optionally, an already-open
SvoReader) and reports whether the expected recorded data/metadata is
present, readable, and internally consistent -- without ever touching a
recorded file. It never repairs, rewrites, resamples, interpolates,
synchronizes, or fuses anything; it only observes what is already on disk
and what the existing readers already loaded, and reports it.

Deliberately reuses what the application already loaded wherever possible
(RecordingSession.frames/radar/events/mti/lidar, SvoReader.timestamps_ns) so
validating a session costs nothing beyond a handful of `Path.exists()` /
`Path.stat()` calls and reading the couple of small JSON files it hasn't
already parsed -- never decoding a camera frame, reconstructing a lidar
scan, or loading a full radar heatmap array.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Status(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


# Worse-is-higher ordering used to roll many individual checks up into one
# section status: any ERROR wins over any WARNING, which wins over
# UNAVAILABLE (an optional feature/dependency simply isn't present -- not a
# data problem), which wins over PASS.
_SEVERITY = {Status.PASS: 0, Status.UNAVAILABLE: 1, Status.WARNING: 2, Status.ERROR: 3}


def worst_status(*statuses: Status) -> Status:
    if not statuses:
        return Status.PASS
    return max(statuses, key=lambda s: _SEVERITY[s])


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""


@dataclass
class SectionReport:
    """One area of the recording (camera, radar, ...): a rolled-up status
    plus every individual check that fed into it, plus machine-readable
    diagnostic fields (counts, timestamps, paths) -- never only a formatted
    string, so a caller can act on the numbers without re-parsing text.
    """

    status: Status = Status.PASS
    checks: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, name: str, status: Status, detail: str = "") -> None:
        self.checks.append(CheckResult(name, status, detail))

    def finalize(self) -> Status:
        self.status = worst_status(*(c.status for c in self.checks)) if self.checks else Status.PASS
        return self.status


@dataclass
class ValidationResult:
    session_id: str
    structure: SectionReport
    manifest: SectionReport
    camera: SectionReport
    frames: SectionReport
    radar: SectionReport
    lidar: SectionReport
    mti: SectionReport
    gnss: SectionReport
    events: SectionReport
    timestamp_summary: SectionReport

    def overall_status(self) -> Status:
        # timestamp_summary is a pure diagnostic (coverage ranges), not a
        # pass/fail signal, so it's deliberately excluded here.
        return worst_status(
            self.structure.status, self.manifest.status, self.camera.status, self.frames.status,
            self.radar.status, self.lidar.status, self.mti.status, self.gnss.status, self.events.status,
        )

    def summary_rows(self) -> list:
        """The six-row overview the UI panel shows (see requirement 14's
        example layout): camera status folds in the frame-metadata check
        since both describe "the camera track" from a viewer's perspective.
        """
        return [
            ("Camera", worst_status(self.camera.status, self.frames.status)),
            ("Radar", self.radar.status),
            ("Lidar", self.lidar.status),
            ("MTi-8", self.mti.status),
            ("GNSS", self.gnss.status),
            ("Events", self.events.status),
        ]

    def all_sections(self) -> list:
        return [
            ("Session structure", self.structure),
            ("Manifest", self.manifest),
            ("Camera / SVO", self.camera),
            ("Camera frame metadata", self.frames),
            ("Radar", self.radar),
            ("Lidar", self.lidar),
            ("MTi-8", self.mti),
            ("GNSS", self.gnss),
            ("Events", self.events),
            ("Timestamp coverage", self.timestamp_summary),
        ]


def _npy_header_readable(path: Path) -> bool:
    """Cheap readability probe for a .npy file: check its magic header
    without loading the array it contains (a radar heatmap here is small,
    but this avoids ever needing to grow that assumption for validation)."""
    try:
        with open(path, "rb") as f:
            magic = f.read(6)
        return magic == b"\x93NUMPY"
    except OSError:
        return False


def _monotonic_nondecreasing(ts: np.ndarray) -> bool:
    return bool(np.all(np.diff(ts) >= 0)) if len(ts) > 1 else True


# --------------------------------------------------------------------------- #
# Section validators. Each is pure and read-only: given already-loaded data
# and/or a session directory Path, it returns a SectionReport and touches
# nothing on disk beyond existence checks and the couple of small files
# (manifest/session_manifest/points JSON, an .npy header) not already
# parsed elsewhere.
# --------------------------------------------------------------------------- #

def validate_session_structure(session_path: Path) -> SectionReport:
    report = SectionReport()
    # (label, path, required) -- "required" means the existing viewer cannot
    # function at all without it; everything else is already treated as
    # optional by the current application (missing sensor -> that panel
    # reports unavailable, per the existing missing-sensor handling).
    entries = [
        ("manifests/dataset_manifest.json", session_path / "manifests" / "dataset_manifest.json", False),
        ("metadata/frames/", session_path / "metadata" / "frames", True),
        ("metadata/radar/", session_path / "metadata" / "radar", False),
        ("metadata/events/", session_path / "metadata" / "events", False),
        ("metadata/sessions/", session_path / "metadata" / "sessions", False),
        ("metadata/gnss/", session_path / "metadata" / "gnss", False),
        ("sensors/", session_path / "sensors", False),
        ("raw/recording.svo2", session_path / "raw" / "recording.svo2", False),
        ("raw/lidar.pcap", session_path / "raw" / "lidar.pcap", False),
        ("raw/lidar_0.json", session_path / "raw" / "lidar_0.json", False),
        ("raw/gnss.nmea.log", session_path / "raw" / "gnss.nmea.log", False),
    ]
    for label, path, required in entries:
        if path.exists():
            report.add(label, Status.PASS, "present")
        else:
            report.add(label, Status.ERROR if required else Status.WARNING, "missing")

    session_manifests = list((session_path / "metadata").glob("session_manifest_*.json")) \
        if (session_path / "metadata").exists() else []
    if session_manifests:
        report.add("metadata/session_manifest_*.json", Status.PASS, f"{len(session_manifests)} found")
    else:
        report.add("metadata/session_manifest_*.json", Status.WARNING, "none found")

    report.finalize()
    return report


# Manifest fields the existing viewer reads with a graceful default/blank
# when absent (see MainWindow._load_session's info_label, _start_playback's
# camera_fps fallback) -- their absence is a WARNING, never an ERROR, and
# this list deliberately does not add any field the app doesn't already use.
_INFORMATIONAL_MANIFEST_FIELDS = ("camera_fps", "resolution", "vehicle", "camera", "imu", "radar", "lidar")


def validate_manifest(session_path: Path) -> SectionReport:
    report = SectionReport()
    manifest_path = session_path / "manifests" / "dataset_manifest.json"
    if not manifest_path.exists():
        # Missing entirely: RecordingSession.load() already tolerates this
        # (defaults to {}), so the viewer still runs -- just with blanks.
        report.add("dataset_manifest.json", Status.WARNING, "file missing; viewer will use defaults")
        report.finalize()
        return report

    try:
        data = json.loads(manifest_path.read_text())
    except Exception as e:
        report.add("dataset_manifest.json", Status.ERROR, f"malformed JSON: {e}")
        report.finalize()
        return report

    if not isinstance(data, dict):
        report.add("dataset_manifest.json", Status.ERROR, "not a JSON object")
        report.finalize()
        return report

    report.add("dataset_manifest.json", Status.PASS, "readable")
    report.info["manifest"] = data
    for name in _INFORMATIONAL_MANIFEST_FIELDS:
        value = data.get(name)
        if value in (None, "", []):
            report.add(f"manifest.{name}", Status.WARNING, "absent or empty (viewer uses a default/blank)")
        else:
            report.add(f"manifest.{name}", Status.PASS, str(value))
    report.finalize()
    return report


def validate_camera(session_path: Path, svo) -> SectionReport:
    """Uses the already-open SvoReader's own index (see svo_reader.py)
    instead of decoding any frame here."""
    report = SectionReport()
    svo_path = session_path / "raw" / "recording.svo2"
    if not svo_path.exists():
        report.add("recording.svo2", Status.WARNING, "missing (camera panel unavailable)")
        report.finalize()
        return report
    report.add("recording.svo2", Status.PASS, "present")

    if svo is None:
        report.add("SVO2 open", Status.UNAVAILABLE, "not opened (ZED SDK unavailable, or open failed)")
        report.finalize()
        return report

    report.add("SVO2 open", Status.PASS, "opened")
    n = int(svo.num_frames)
    report.info["frame_count"] = n
    if n == 0:
        report.add("SVO frame count", Status.ERROR, "0 frames")
        report.finalize()
        return report

    report.add("SVO frame count", Status.PASS, str(n))
    ts = svo.timestamps_ns
    report.info["first_timestamp_ns"] = int(ts[0])
    report.info["last_timestamp_ns"] = int(ts[-1])
    ordered = _monotonic_nondecreasing(ts)
    report.add("SVO timestamp ordering", Status.PASS if ordered else Status.WARNING,
                "non-decreasing" if ordered else "not sorted ascending")
    report.add("SVO frame index available", Status.PASS, f"{n} entries")
    report.add("ZED onboard IMU", Status.PASS if svo.has_zed_imu else Status.WARNING,
                "available" if svo.has_zed_imu else "not recorded in this SVO2")
    report.finalize()
    return report


def validate_frames(frames_df) -> SectionReport:
    """Inspects RecordingSession.frames as already loaded; never modifies it."""
    report = SectionReport()
    n = len(frames_df)
    report.info["row_count"] = n
    if n == 0:
        report.add("frame rows", Status.ERROR, "no camera frame metadata rows")
        report.finalize()
        return report

    report.add("frame rows", Status.PASS, str(n))
    ts = frames_df["timestamp_ns"].to_numpy()
    finite = np.isfinite(ts.astype(np.float64))
    if finite.all():
        report.add("frame timestamp validity", Status.PASS, "all finite")
    else:
        report.add("frame timestamp validity", Status.ERROR, f"{int((~finite).sum())} non-finite timestamp(s)")

    report.info["first_timestamp_ns"] = int(ts[0])
    report.info["last_timestamp_ns"] = int(ts[-1])
    # RecordingSession already sorts by timestamp on load, so this mainly
    # guards against a future regression rather than catching out-of-order
    # raw files (that ordering isn't observable any more once loaded).
    ordered = _monotonic_nondecreasing(ts)
    report.add("frame timestamp ordering", Status.PASS if ordered else Status.WARNING,
                "non-decreasing" if ordered else "not sorted ascending")

    duplicates = n - len(np.unique(ts))
    report.add("duplicate frame timestamps", Status.PASS if duplicates == 0 else Status.WARNING,
                "none" if duplicates == 0 else f"{duplicates} duplicate value(s)")
    report.finalize()
    return report


def validate_radar(session) -> SectionReport:
    report = SectionReport()
    radar_df = session.radar
    n = len(radar_df)
    report.info["row_count"] = n
    if n == 0:
        report.add("radar metadata rows", Status.WARNING, "no radar metadata (radar unavailable for this session)")
        report.finalize()
        return report

    report.add("radar metadata rows", Status.PASS, str(n))
    ts = radar_df["timestamp_ns"].to_numpy()
    report.info["first_timestamp_ns"] = int(ts[0])
    report.info["last_timestamp_ns"] = int(ts[-1])
    finite = np.isfinite(ts.astype(np.float64))
    report.add("radar timestamp validity", Status.PASS if finite.all() else Status.ERROR,
                "all finite" if finite.all() else f"{int((~finite).sum())} non-finite timestamp(s)")
    ordered = _monotonic_nondecreasing(ts)
    report.add("radar timestamp ordering", Status.PASS if ordered else Status.WARNING,
                "non-decreasing" if ordered else "not sorted ascending")

    missing_heatmap = missing_points = unreadable_heatmap = unreadable_points = 0
    for _, row in radar_df.iterrows():
        hp = row.get("heatmap_path")
        if hp:
            full = session.path / hp
            if not full.exists():
                missing_heatmap += 1
            elif not _npy_header_readable(full):
                unreadable_heatmap += 1
        pp = row.get("points_path")
        if pp:
            full = session.path / pp
            if not full.exists():
                missing_points += 1
            else:
                try:
                    json.loads(full.read_text())
                except Exception:
                    unreadable_points += 1

    report.info.update({
        "missing_heatmap_files": missing_heatmap,
        "missing_points_files": missing_points,
        "unreadable_heatmap_files": unreadable_heatmap,
        "unreadable_points_files": unreadable_points,
    })
    if missing_heatmap or missing_points:
        report.add("radar referenced files present", Status.ERROR,
                    f"{missing_heatmap} heatmap file(s), {missing_points} points file(s) missing")
    else:
        report.add("radar referenced files present", Status.PASS, "all present")
    if unreadable_heatmap or unreadable_points:
        report.add("radar referenced files readable", Status.ERROR,
                    f"{unreadable_heatmap} heatmap file(s), {unreadable_points} points file(s) unreadable")
    else:
        report.add("radar referenced files readable", Status.PASS, "all readable")
    report.finalize()
    return report


def validate_lidar(session) -> SectionReport:
    """Reuses LidarReader's own already-parsed state (session.lidar /
    session.lidar_error, built once at session load -- see session.py);
    never re-parses the pcap here."""
    report = SectionReport()
    pcap_path = session.path / "raw" / "lidar.pcap"
    meta_path = session.path / "raw" / "lidar_0.json"
    report.add("lidar.pcap", Status.PASS if pcap_path.exists() else Status.WARNING,
                "present" if pcap_path.exists() else "missing")
    report.add("lidar_0.json", Status.PASS if meta_path.exists() else Status.WARNING,
                "present" if meta_path.exists() else "missing")

    if session.lidar is not None:
        lidar = session.lidar
        n = lidar.num_scans
        report.info["scan_count"] = n
        report.add("lidar reader opened", Status.PASS, "opened")
        if n > 0:
            ts = lidar.timestamps_ns
            report.info["first_timestamp_ns"] = int(ts[0])
            report.info["last_timestamp_ns"] = int(ts[-1])
            report.add("lidar scan count", Status.PASS, str(n))
            ordered = _monotonic_nondecreasing(ts)
            report.add("lidar timestamp ordering", Status.PASS if ordered else Status.WARNING,
                        "non-decreasing" if ordered else "not sorted ascending")
            report.add("lidar point cache", Status.PASS, f"{len(lidar.points)} scan(s) cached in memory")
        else:
            report.add("lidar scan count", Status.WARNING, "0 scans")
    elif session.lidar_error:
        report.add("lidar reader opened", Status.ERROR, session.lidar_error)
    elif pcap_path.exists() and meta_path.exists():
        try:
            import ouster.sdk  # noqa: F401
        except ImportError:
            report.add("lidar reader opened", Status.UNAVAILABLE, "ouster-sdk is not installed")
        else:
            report.add("lidar reader opened", Status.WARNING, "files present but not loaded")
    else:
        report.add("lidar reader opened", Status.WARNING, "no lidar files for this session")
    report.finalize()
    return report


def validate_mti(session) -> SectionReport:
    report = SectionReport()
    sensors_dir = session.path / "sensors"
    h5_files = list(sensors_dir.glob("*.h5")) if sensors_dir.exists() else []
    report.add("sensors/*.h5", Status.PASS if h5_files else Status.WARNING,
                f"{len(h5_files)} file(s)" if h5_files else "none found")

    mti = session.mti
    if mti is None:
        report.add("MTi data loaded", Status.WARNING, "not available")
        report.finalize()
        return report

    report.add("MTi data loaded", Status.PASS, "loaded")
    n = mti.n
    report.info["sample_count"] = n
    if n == 0:
        report.add("MTi sample count", Status.WARNING, "0 samples")
        report.finalize()
        return report

    report.add("MTi sample count", Status.PASS, str(n))
    ts = mti.timestamp_ns
    report.info["first_timestamp_ns"] = int(ts[0])
    report.info["last_timestamp_ns"] = int(ts[-1])
    finite = np.isfinite(ts.astype(np.float64))
    report.add("MTi timestamp validity", Status.PASS if finite.all() else Status.ERROR,
                "all finite" if finite.all() else f"{int((~finite).sum())} non-finite timestamp(s)")
    ordered = _monotonic_nondecreasing(ts)
    report.add("MTi timestamp ordering", Status.PASS if ordered else Status.WARNING,
                "non-decreasing" if ordered else "not sorted ascending")

    report.add("acceleration", Status.PASS if mti.acceleration.shape[0] == n else Status.ERROR,
                str(mti.acceleration.shape))
    report.add("angular_velocity", Status.PASS if mti.angular_velocity.shape[0] == n else Status.ERROR,
                str(mti.angular_velocity.shape))
    report.add("quaternion", Status.PASS if mti.quaternion_raw.shape[0] == n else Status.ERROR,
                str(mti.quaternion_raw.shape))

    has_fix = _has_valid_fix(mti.latitude, mti.longitude)
    report.info["has_valid_fix"] = has_fix
    report.add("fused position (lat/lon)", Status.PASS if has_fix else Status.WARNING,
                "at least one valid fix" if has_fix else "no valid fix in this recording")
    report.finalize()
    return report


def _has_valid_fix(latitude: np.ndarray, longitude: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(latitude) & np.isfinite(longitude) & (latitude != 0) & (longitude != 0)))


def validate_gnss(session) -> SectionReport:
    """Distinguishes an empty/missing raw NMEA log (expected on this
    hardware -- see SYSTEM_FUNCTIONS.md) from GNSS actually being
    unavailable to the viewer, which is instead driven by MTi fused
    position -- exactly the existing behavior in MainWindow._plot_gnss_track.
    """
    report = SectionReport()
    nmea_path = session.path / "raw" / "gnss.nmea.log"
    nmea_exists = nmea_path.exists()
    nmea_size = nmea_path.stat().st_size if nmea_exists else 0
    report.info["raw_nmea_bytes"] = nmea_size

    if not nmea_exists:
        report.add("raw NMEA log", Status.WARNING, "file missing")
    elif nmea_size == 0:
        report.add("raw NMEA log", Status.WARNING, "empty")
    else:
        report.add("raw NMEA log", Status.PASS, f"{nmea_size} bytes")

    mti = session.mti
    fused_available = mti is not None and mti.n > 0 and _has_valid_fix(mti.latitude, mti.longitude)
    report.info["mti_fused_available"] = fused_available
    report.add("MTi fused position", Status.PASS if fused_available else Status.WARNING,
                "available" if fused_available else "no valid fix")

    if fused_available:
        report.add("GNSS viewer availability", Status.PASS, "available via MTi fused position")
    elif nmea_exists and nmea_size > 0:
        report.add("GNSS viewer availability", Status.WARNING,
                    "raw NMEA present but the viewer reads GNSS via MTi fused position, which has no valid fix")
    else:
        report.add("GNSS viewer availability", Status.WARNING,
                    "unavailable (no raw fix and no MTi fused position)")
    report.finalize()
    return report


def validate_events(session) -> SectionReport:
    report = SectionReport()
    events = session.events
    n = len(events)
    report.info["event_count"] = n
    if n == 0:
        report.add("event rows", Status.WARNING, "no events recorded")
        report.finalize()
        return report

    report.add("event rows", Status.PASS, str(n))
    ts = events["timestamp_ns"].to_numpy()
    report.info["earliest_event_ns"] = int(ts.min())
    report.info["latest_event_ns"] = int(ts.max())
    types = sorted(str(t) for t in events["event_type"].unique().tolist())
    report.info["event_types"] = types

    # Surface drop/counter events verbatim -- never interpret why a sensor
    # dropped frames, just report that the existing metadata records it.
    drop_mask = events["event_type"].astype(str).str.contains("DROP", case=False, na=False)
    drop_rows = events[drop_mask]
    drops = [
        {"event_type": r["event_type"], "description": r["description"]}
        for _, r in drop_rows.iterrows()
    ]
    report.info["drop_events"] = drops
    if drops:
        report.add("sensor drop events", Status.WARNING,
                    f"{len(drops)} drop event(s) recorded: " + "; ".join(f"{d['event_type']}: {d['description']}" for d in drops))
    else:
        report.add("sensor drop events", Status.PASS, "none recorded")
    report.finalize()
    return report


def validate_timestamp_coverage(session, svo) -> SectionReport:
    """Diagnostic only: reports each sensor's timestamp range relative to
    the camera's own start, in seconds. Never used to alter synchronization.
    """
    report = SectionReport()
    frames = session.frames
    if frames.empty:
        report.add("camera", Status.ERROR, "no camera frames")
        report.finalize()
        return report

    cam_ts = frames["timestamp_ns"].to_numpy()
    t0 = int(cam_ts[0])

    def rel_s(ns: int) -> float:
        return (int(ns) - t0) / 1e9

    def add_range(name: str, ts_array, unavailable_detail: str) -> None:
        if ts_array is None or len(ts_array) == 0:
            report.add(name, Status.UNAVAILABLE, unavailable_detail)
            report.info[f"{name.lower()}_range_s"] = None
            return
        lo, hi = rel_s(ts_array[0]), rel_s(ts_array[-1])
        report.info[f"{name.lower()}_range_s"] = (lo, hi)
        report.add(name, Status.PASS, f"{lo:.3f} s -> {hi:.3f} s")

    add_range("camera", cam_ts, "no camera frames")
    radar_ts = session.radar["timestamp_ns"].to_numpy() if not session.radar.empty else None
    add_range("radar", radar_ts, "no radar metadata")
    add_range("lidar", session.lidar.timestamps_ns if (session.lidar is not None and session.lidar.num_scans > 0) else None,
               "lidar unavailable")
    add_range("mti", session.mti.timestamp_ns if (session.mti is not None and session.mti.n > 0) else None,
               "MTi unavailable")
    add_range("svo", svo.timestamps_ns if (svo is not None and svo.num_frames > 0) else None,
               "camera not opened")
    report.finalize()
    return report


class DatasetIntegrityValidator:
    """Validates one RecordingSession (optionally with its already-open
    SvoReader), purely by reading -- never by writing, repairing, or
    regenerating anything.

    Accepts the real RecordingSession, or any object exposing the same
    read-only surface it needs (`.path`, `.session_id`, `.frames`, `.radar`,
    `.events`, `.mti`, `.lidar`, `.lidar_error`) -- "an equivalent session
    description" per the design brief -- which is what the tests use to
    exercise failure paths without needing a real recording on disk.
    """

    def __init__(self, session, svo=None):
        self.session = session
        self.svo = svo

    def validate(self) -> ValidationResult:
        session = self.session
        result = ValidationResult(
            session_id=session.session_id,
            structure=validate_session_structure(session.path),
            manifest=validate_manifest(session.path),
            camera=validate_camera(session.path, self.svo),
            frames=validate_frames(session.frames),
            radar=validate_radar(session),
            lidar=validate_lidar(session),
            mti=validate_mti(session),
            gnss=validate_gnss(session),
            events=validate_events(session),
            timestamp_summary=validate_timestamp_coverage(session, self.svo),
        )
        logger.info(
            "DatasetIntegrityValidator: session=%s overall=%s (%s)",
            result.session_id, result.overall_status().value,
            ", ".join(f"{name}={status.value}" for name, status in result.summary_rows()),
        )
        return result
