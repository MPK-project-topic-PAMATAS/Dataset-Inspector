"""Tests for the read-only, position-aware Sensor Health / Timeline
Diagnostics model (sensor_health.py) and its wiring into MainWindow.

Everything here uses small fakes for RecordingSession/SyncIndex/SvoReader/
ValidationResult so behavior can be checked precisely and cheaply, without
a real recording. A dedicated real-dataset check lives in
test_sensor_health_real_dataset (skipped if the recording isn't present).
"""
from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensor_health import (
    SensorHealthModel,
    SensorState,
    SyncThresholds,
)
from sync_index import MISSING


class FakeSyncIndex:
    """A SyncIndex stand-in with fixed, explicit mappings -- so a test can
    assert the health model used exactly that mapping, not a fresh search."""

    def __init__(self, radar=0, lidar=0, mti=0, svo=0):
        self._radar, self._lidar, self._mti, self._svo = radar, lidar, mti, svo

    def radar_for(self, i):
        return self._radar

    def lidar_for(self, i):
        return self._lidar

    def mti_for(self, i):
        return self._mti

    def svo_for(self, i):
        return self._svo


class FakeMti:
    def __init__(self, timestamp_ns, latitude=None, longitude=None):
        self.timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64)
        n = len(self.timestamp_ns)
        self.latitude = np.asarray(latitude if latitude is not None else np.zeros(n))
        self.longitude = np.asarray(longitude if longitude is not None else np.zeros(n))

    @property
    def n(self):
        return len(self.timestamp_ns)


class FakeLidar:
    def __init__(self, timestamps_ns):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)

    @property
    def num_scans(self):
        return len(self.timestamps_ns)


class FakeSvo:
    def __init__(self, timestamps_ns, has_zed_imu=True):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self.has_zed_imu = has_zed_imu


class FakeSession:
    def __init__(self, path=Path("/nonexistent"), frames=None, radar=None, events=None, mti=None, lidar=None):
        self.path = path
        self.session_id = "fake"
        self.frames = frames if frames is not None else pd.DataFrame({"timestamp_ns": [1_000_000_000]})
        self.radar = radar if radar is not None else pd.DataFrame()
        self.events = events if events is not None else pd.DataFrame()
        self.mti = mti
        self.lidar = lidar


class FakeSectionReport:
    def __init__(self, checks=None, info=None):
        self.checks = checks or []
        self.info = info or {}


class FakeCheck:
    def __init__(self, status):
        class _S:
            value = status
        self.status = _S()


class FakeValidationResult:
    def __init__(self, mti_info=None, lidar_checks=None):
        self.mti = FakeSectionReport(info=mti_info or {})
        self.lidar = FakeSectionReport(checks=lidar_checks or [])


def _make_model(**kwargs):
    session = kwargs.pop("session", FakeSession())
    sync_index = kwargs.pop("sync_index", FakeSyncIndex())
    svo = kwargs.pop("svo", None)
    validation_result = kwargs.pop("validation_result", None)
    thresholds = kwargs.pop("thresholds", None)
    return SensorHealthModel(session, sync_index, svo, validation_result, thresholds)


# --------------------------------------------------------------------------- #
# Camera timestamp / frame index.
# --------------------------------------------------------------------------- #

def test_camera_timestamp_and_frame_index_are_displayed_correctly():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000, 1_100_000_000, 1_200_000_000]})
    model = _make_model(session=FakeSession(frames=frames))
    snap = model.snapshot(1)
    assert snap.camera_frame_index == 1
    assert snap.camera_timestamp_ns == 1_100_000_000
    assert snap.camera.state == SensorState.AVAILABLE
    assert snap.camera.delta_ns == 0


# --------------------------------------------------------------------------- #
# Sample index / timestamp / signed delta come from SyncIndex + the selected
# recorded sample, not from a fresh calculation.
# --------------------------------------------------------------------------- #

def test_radar_sample_and_delta_come_from_sync_index_selected_row():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
    radar = pd.DataFrame({"timestamp_ns": [900_000_000, 1_005_000_000], "frame_id": [7, 8]})
    model = _make_model(session=FakeSession(frames=frames, radar=radar), sync_index=FakeSyncIndex(radar=1))

    diag = model.snapshot(0).radar
    assert diag.sample_index == 8  # the row FakeSyncIndex.radar_for(0)=1 points at, i.e. frame_id 8
    assert diag.sensor_timestamp_ns == 1_005_000_000
    assert diag.delta_ns == 1_005_000_000 - 1_000_000_000  # sensor - camera, signed
    assert diag.delta_ms == pytest.approx(5.0)
    assert diag.state == SensorState.AVAILABLE


def test_lidar_sample_and_delta_come_from_sync_index_selected_scan():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
    lidar = FakeLidar([500_000_000, 995_000_000])
    model = _make_model(session=FakeSession(frames=frames, lidar=lidar), sync_index=FakeSyncIndex(lidar=1))

    diag = model.snapshot(0).lidar
    assert diag.sample_index == 1
    assert diag.sensor_timestamp_ns == 995_000_000
    assert diag.delta_ns == 995_000_000 - 1_000_000_000
    assert diag.delta_ms == pytest.approx(-5.0)


def test_mti_sample_and_delta_come_from_sync_index_selected_sample():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
    mti = FakeMti([999_000_000, 1_000_800_000])
    model = _make_model(session=FakeSession(frames=frames, mti=mti), sync_index=FakeSyncIndex(mti=1))

    diag = model.snapshot(0).mti
    assert diag.sample_index == 1
    assert diag.sensor_timestamp_ns == 1_000_800_000
    assert diag.delta_ns == 800_000
    assert diag.delta_ms == pytest.approx(0.8)


def test_zed_imu_sample_and_delta_come_from_sync_index_svo_mapping():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
    svo = FakeSvo([998_500_000, 1_002_000_000], has_zed_imu=True)
    model = _make_model(session=FakeSession(frames=frames), sync_index=FakeSyncIndex(svo=1), svo=svo)

    diag = model.snapshot(0).zed_imu
    assert diag.sample_index == 1
    assert diag.sensor_timestamp_ns == 1_002_000_000
    assert diag.delta_ns == 2_000_000
    assert diag.delta_ms == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Missing / unavailable / no-fix states.
# --------------------------------------------------------------------------- #

def test_missing_radar_data_is_no_data():
    model = _make_model(session=FakeSession(radar=pd.DataFrame()))
    assert model.snapshot(0).radar.state == SensorState.NO_DATA


def test_missing_sync_mapping_is_no_data():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
    radar = pd.DataFrame({"timestamp_ns": [900_000_000], "frame_id": [1]})
    model = _make_model(session=FakeSession(frames=frames, radar=radar), sync_index=FakeSyncIndex(radar=MISSING))
    assert model.snapshot(0).radar.state == SensorState.NO_DATA


def test_lidar_dependency_unavailable_is_unavailable_not_no_data():
    validation_result = FakeValidationResult(lidar_checks=[FakeCheck("UNAVAILABLE")])
    model = _make_model(session=FakeSession(lidar=None), validation_result=validation_result)
    assert model.snapshot(0).lidar.state == SensorState.UNAVAILABLE


def test_lidar_missing_without_unavailable_dependency_flag_is_no_data():
    validation_result = FakeValidationResult(lidar_checks=[FakeCheck("PASS")])
    model = _make_model(session=FakeSession(lidar=None), validation_result=validation_result)
    assert model.snapshot(0).lidar.state == SensorState.NO_DATA


def test_missing_mti_hdf5_is_no_data():
    model = _make_model(session=FakeSession(mti=None))
    assert model.snapshot(0).mti.state == SensorState.NO_DATA


def test_missing_gnss_fix_is_no_fix():
    mti = FakeMti([999_000_000], latitude=[0.0], longitude=[0.0])  # loaded, but no valid fix
    model = _make_model(session=FakeSession(mti=mti))
    assert model.snapshot(0).gnss.state == SensorState.NO_FIX


def test_gnss_available_via_mti_fusion_reuses_validators_flag():
    mti = FakeMti([999_000_000], latitude=[0.0], longitude=[0.0])  # array itself has no fix...
    validation_result = FakeValidationResult(mti_info={"has_valid_fix": True})  # ...but validator says fix exists
    model = _make_model(session=FakeSession(mti=mti), validation_result=validation_result)
    # The validator's already-computed flag is reused rather than recomputed,
    # proving GNSS availability rides on the exact same determination
    # DatasetIntegrityValidator already made.
    assert model.snapshot(0).gnss.state == SensorState.AVAILABLE


def test_gnss_no_mti_at_all_is_no_data():
    model = _make_model(session=FakeSession(mti=None))
    assert model.snapshot(0).gnss.state == SensorState.NO_DATA


# --------------------------------------------------------------------------- #
# Recorded drop events -> DROPPED state and timeline diagnostics.
# --------------------------------------------------------------------------- #

def test_recorded_radar_drop_event_is_exposed_as_timeline_diagnostic():
    events = pd.DataFrame({
        "timestamp_ns": [5_000_000_000],
        "event_type": ["RADAR_DROPPED_FRAMES"],
        "description": ["dropped=15"],
    })
    model = _make_model(session=FakeSession(events=events))
    assert len(model.drop_events) == 1
    assert model.drop_events[0].event_type == "RADAR_DROPPED_FRAMES"
    assert model.drop_events[0].description == "dropped=15"
    assert model.drop_events[0].timestamp_ns == 5_000_000_000


def test_recorded_lidar_drop_event_is_exposed_as_timeline_diagnostic():
    events = pd.DataFrame({
        "timestamp_ns": [6_000_000_000],
        "event_type": ["LIDAR_DROPPED_FRAMES"],
        "description": ["dropped=41"],
    })
    model = _make_model(session=FakeSession(events=events))
    assert len(model.drop_events) == 1
    assert model.drop_events[0].event_type == "LIDAR_DROPPED_FRAMES"


def test_radar_sample_exactly_matching_a_recorded_drop_event_timestamp_is_dropped():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
    radar = pd.DataFrame({"timestamp_ns": [1_000_000_000], "frame_id": [9]})  # same ts as the event below
    events = pd.DataFrame({
        "timestamp_ns": [1_000_000_000],
        "event_type": ["RADAR_DROPPED_FRAMES"],
        "description": ["dropped=15"],
    })
    model = _make_model(session=FakeSession(frames=frames, radar=radar, events=events))
    assert model.snapshot(0).radar.state == SensorState.DROPPED


def test_non_drop_events_are_not_collected_as_drop_events():
    events = pd.DataFrame({
        "timestamp_ns": [1, 2],
        "event_type": ["RECORDING_START", "MTI_COUNTERS"],
        "description": ["x", "y"],
    })
    model = _make_model(session=FakeSession(events=events))
    assert model.drop_events == []


# --------------------------------------------------------------------------- #
# Thresholds: display-only, never alter SyncIndex selection or sensor data.
# --------------------------------------------------------------------------- #

def test_threshold_classification_boundaries():
    thresholds = SyncThresholds(good_ms=20.0, warning_ms=60.0)
    assert thresholds.classify(10.0) == "GOOD"
    assert thresholds.classify(20.0) == "GOOD"
    assert thresholds.classify(20.1) == "WARNING"
    assert thresholds.classify(60.0) == "WARNING"
    assert thresholds.classify(60.1) == "STALE"


def test_changing_thresholds_only_changes_displayed_classification():
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
    radar = pd.DataFrame({"timestamp_ns": [1_050_000_000], "frame_id": [1]})  # 50ms ahead of camera
    session = FakeSession(frames=frames, radar=radar)
    sync_index = FakeSyncIndex(radar=0)

    lenient = _make_model(session=session, sync_index=sync_index, thresholds=SyncThresholds(good_ms=100, warning_ms=200))
    strict = _make_model(session=session, sync_index=sync_index, thresholds=SyncThresholds(good_ms=5, warning_ms=10))

    diag_lenient = lenient.snapshot(0).radar
    diag_strict = strict.snapshot(0).radar

    # Same selected sample, same timestamp, same delta -- only the label differs.
    assert diag_lenient.sample_index == diag_strict.sample_index == 1
    assert diag_lenient.sensor_timestamp_ns == diag_strict.sensor_timestamp_ns == 1_050_000_000
    assert diag_lenient.delta_ns == diag_strict.delta_ns == 50_000_000
    assert diag_lenient.sync_class == "GOOD"
    assert diag_lenient.state == SensorState.AVAILABLE
    assert diag_strict.sync_class == "STALE"
    assert diag_strict.state == SensorState.STALE


def test_changing_thresholds_does_not_alter_sync_index_or_session_data():
    session = FakeSession()
    sync_index = FakeSyncIndex(radar=3, lidar=4, mti=5, svo=6)
    before = (sync_index.radar_for(0), sync_index.lidar_for(0), sync_index.mti_for(0), sync_index.svo_for(0))
    before_frames = session.frames.copy()

    _make_model(session=session, sync_index=sync_index, thresholds=SyncThresholds(good_ms=1, warning_ms=2))
    _make_model(session=session, sync_index=sync_index, thresholds=SyncThresholds(good_ms=999, warning_ms=9999))

    after = (sync_index.radar_for(0), sync_index.lidar_for(0), sync_index.mti_for(0), sync_index.svo_for(0))
    assert before == after
    pd.testing.assert_frame_equal(before_frames, session.frames)


# --------------------------------------------------------------------------- #
# No disk I/O on the diagnostic update path.
# --------------------------------------------------------------------------- #

def test_snapshot_performs_no_disk_io(monkeypatch):
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000, 1_100_000_000]})
    radar = pd.DataFrame({"timestamp_ns": [1_005_000_000], "frame_id": [1]})
    mti = FakeMti([1_000_500_000])
    lidar = FakeLidar([999_500_000])
    svo = FakeSvo([1_000_200_000])
    session = FakeSession(frames=frames, radar=radar, mti=mti, lidar=lidar)
    model = _make_model(session=session, svo=svo)

    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("sensor health snapshot must not perform disk I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        for i in range(len(frames)):
            model.snapshot(i)
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


# --------------------------------------------------------------------------- #
# MainWindow-level: seeking semantics, and no stale diagnostic display.
# --------------------------------------------------------------------------- #

@pytest.fixture
def window(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)
    yield win
    win.close()


def test_seek_to_timestamp_matches_existing_event_seek_semantics(window):
    frames = pd.DataFrame({"timestamp_ns": [0, 100, 200, 300]})
    window.session = FakeSession(frames=frames)

    window._seek_to_timestamp(150)  # nearest frame at/after 150 -> index 2 (ts=200)
    assert window.current_index == 2

    window._seek_to_timestamp(200)  # exact match -> index 2
    assert window.current_index == 2

    window._seek_to_timestamp(1000)  # past the end -> clamps to the last frame
    assert window.current_index == 3


def test_event_list_double_click_uses_the_same_seek_path(window, monkeypatch):
    frames = pd.DataFrame({"timestamp_ns": [0, 100, 200, 300]})
    window.session = FakeSession(frames=frames)

    seen = []
    monkeypatch.setattr(window, "_seek_to_timestamp", lambda ts: seen.append(ts))

    class FakeItem:
        def data(self, role):
            return 200

    window._seek_to_event(FakeItem())
    assert seen == [200]


def test_marker_click_uses_the_same_seek_path_as_event_list(window, monkeypatch):
    seen = []
    monkeypatch.setattr(window, "_seek_to_timestamp", lambda ts: seen.append(ts))
    window.event_marker_strip.markerClicked.emit(123456789)
    assert seen == [123456789]


def test_sensor_health_snapshot_never_lags_behind_current_index(window):
    """Since sensor-health snapshots are computed synchronously per seek
    (never waiting on SensorWorker), rapid re-seeking can never leave a
    stale snapshot on display: after seeking a -> b -> c, the displayed
    snapshot always reflects c, never a or b."""
    frames = pd.DataFrame({"timestamp_ns": np.arange(0, 1000, 10, dtype=np.int64)})
    session = FakeSession(frames=frames)  # no radar/lidar/mti data -> nothing for dispatch to fetch
    window.session = session
    window.sync_index = FakeSyncIndex(radar=MISSING, lidar=MISSING, mti=MISSING, svo=MISSING)
    window.sensor_health = SensorHealthModel(session, window.sync_index)

    for idx in (5, 40, 3, 77, 12):
        window._seek(idx)

    assert window._current_health_snapshot.camera_frame_index == 12


# --------------------------------------------------------------------------- #
# Real dataset: the recording known (from the Dataset Integrity report) to
# have RADAR_DROPPED_FRAMES dropped=15 and LIDAR_DROPPED_FRAMES dropped=41.
# --------------------------------------------------------------------------- #

_DATASET_ROOT = Path(__file__).parent.parent / "output_multi-record" / "output"
_REAL_SESSION_DIR = _DATASET_ROOT / "20260914_070614"


@pytest.mark.skipif(not _REAL_SESSION_DIR.exists(), reason="real recorded session not available")
def test_real_dataset_exposes_recorded_drop_events():
    from session import RecordingSession

    session = RecordingSession.load(_REAL_SESSION_DIR)
    model = SensorHealthModel(session, FakeSyncIndex())

    drops = {ev.event_type: ev.description for ev in model.drop_events}
    assert drops.get("RADAR_DROPPED_FRAMES") == "dropped=15"
    assert drops.get("LIDAR_DROPPED_FRAMES") == "dropped=41"


@pytest.mark.skipif(not _REAL_SESSION_DIR.exists(), reason="real recorded session not available")
def test_real_dataset_gnss_no_fix_matches_dataset_integrity_validator():
    from session import RecordingSession
    from validator import DatasetIntegrityValidator

    session = RecordingSession.load(_REAL_SESSION_DIR)
    validation_result = DatasetIntegrityValidator(session, svo=None).validate()
    model = SensorHealthModel(session, FakeSyncIndex(), validation_result=validation_result)

    # This recording is known to have MTi data but no valid GNSS fix.
    assert model.snapshot(0).mti.state != SensorState.NO_DATA
    assert model.snapshot(0).gnss.state == SensorState.NO_FIX
