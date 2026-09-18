"""Tests for the read-only Dataset Integrity Validator (validator.py).

These build small synthetic session directories under pytest's tmp_path
and lightweight fake session/svo objects -- shaped like RecordingSession/
SvoReader but without needing a real recording -- so every failure path
(missing file, malformed JSON, corrupt heatmap, invalid timestamps, ...)
can be exercised directly and cheaply. Every test also checks that running
the validator left the files it touched completely unchanged: it must only
ever read.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validator import (
    DatasetIntegrityValidator,
    SectionReport,
    Status,
    ValidationResult,
    validate_camera,
    validate_events,
    validate_frames,
    validate_gnss,
    validate_lidar,
    validate_manifest,
    validate_mti,
    validate_radar,
    validate_session_structure,
    validate_timestamp_coverage,
    worst_status,
)


class FakeMti:
    def __init__(self, timestamp_ns, latitude=None, longitude=None):
        n = len(timestamp_ns)
        self.timestamp_ns = np.asarray(timestamp_ns, dtype=np.float64 if any(
            not np.isfinite(t) for t in timestamp_ns) else np.int64)
        self.acceleration = np.zeros((n, 3), dtype=np.float32)
        self.angular_velocity = np.zeros((n, 3), dtype=np.float32)
        self.quaternion_raw = np.zeros((n, 4), dtype=np.float32)
        self.latitude = np.asarray(latitude if latitude is not None else np.zeros(n))
        self.longitude = np.asarray(longitude if longitude is not None else np.zeros(n))

    @property
    def n(self):
        return len(self.timestamp_ns)


class FakeLidar:
    def __init__(self, timestamps_ns, points=None):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self.points = points if points is not None else [np.zeros((0, 3))] * len(self.timestamps_ns)

    @property
    def num_scans(self):
        return len(self.timestamps_ns)


class FakeSvo:
    def __init__(self, timestamps_ns, has_zed_imu=True):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self.num_frames = len(self.timestamps_ns)
        self.has_zed_imu = has_zed_imu


class FakeSession:
    """Shaped like RecordingSession -- an "equivalent session description"
    per validator.py's docstring -- pointed at a real tmp_path directory so
    file-existence checks exercise the real filesystem.
    """

    def __init__(self, path, frames=None, radar=None, events=None, mti=None, lidar=None, lidar_error=None):
        self.path = Path(path)
        self.session_id = self.path.name
        self.frames = frames if frames is not None else pd.DataFrame()
        self.radar = radar if radar is not None else pd.DataFrame()
        self.events = events if events is not None else pd.DataFrame()
        self.mti = mti
        self.lidar = lidar
        self.lidar_error = lidar_error


def _touch(path: Path, content: bytes = b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _npy_bytes(shape=(2, 2)) -> bytes:
    import io
    buf = io.BytesIO()
    np.save(buf, np.zeros(shape, dtype=np.uint16))
    return buf.getvalue()


def _make_full_valid_session_dir(tmp_path) -> Path:
    """A structurally and internally consistent synthetic session, used as
    the "everything passes" baseline."""
    root = tmp_path / "20260101_000000"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "dataset_manifest.json").write_text(json.dumps({
        "camera_fps": 30.0, "resolution": [1280, 720], "vehicle": "test-vehicle",
        "camera": "ZED 2", "imu": "MTi-8", "radar": "TI AWR1843BOOST", "lidar": "Ouster OS-0-64",
    }))
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / "session_manifest_20260101_000000.json").write_text("{}")
    for d in ("frames", "radar", "events", "sessions", "gnss"):
        (root / "metadata" / d).mkdir(parents=True)
    (root / "sensors").mkdir(parents=True)
    _touch(root / "sensors" / "dummy.h5")
    _touch(root / "raw" / "recording.svo2")
    _touch(root / "raw" / "lidar.pcap")
    _touch(root / "raw" / "lidar_0.json", b"{}")
    _touch(root / "raw" / "gnss.nmea.log", b"$GPGGA,...\n")

    heatmap_path = root / "radar" / "heatmap" / "000.npy"
    points_path = root / "radar" / "points" / "000.json"
    _touch(heatmap_path, _npy_bytes())
    _touch(points_path, json.dumps({"points": []}).encode())

    frames = pd.DataFrame({"timestamp_ns": np.arange(0, 500, 100, dtype=np.int64)})
    radar = pd.DataFrame({
        "timestamp_ns": np.array([50, 150], dtype=np.int64),
        "heatmap_path": [str(heatmap_path.relative_to(root))] * 2,
        "points_path": [str(points_path.relative_to(root))] * 2,
    })
    events = pd.DataFrame({
        "timestamp_ns": np.array([0, 400], dtype=np.int64),
        "event_type": ["RECORDING_START", "RECORDING_STOP"],
        "description": ["session=x", "session=x"],
    })
    mti = FakeMti(np.arange(0, 500, 10, dtype=np.int64), latitude=np.full(50, 1.0), longitude=np.full(50, 2.0))
    lidar = FakeLidar(np.array([10, 200, 400], dtype=np.int64))

    session = FakeSession(root, frames=frames, radar=radar, events=events, mti=mti, lidar=lidar)
    svo = FakeSvo(np.arange(0, 500, 100, dtype=np.int64))
    return session, svo


def _snapshot(paths):
    return {p: p.read_bytes() for p in paths if p.exists()}


# --------------------------------------------------------------------------- #
# 1. A fully valid synthetic session.
# --------------------------------------------------------------------------- #

def test_fully_valid_synthetic_session_is_all_pass(tmp_path):
    session, svo = _make_full_valid_session_dir(tmp_path)
    result = DatasetIntegrityValidator(session, svo).validate()

    assert result.overall_status() == Status.PASS
    for name, status in result.summary_rows():
        assert status == Status.PASS, f"{name} was {status}"


def test_validation_does_not_modify_any_file(tmp_path):
    session, svo = _make_full_valid_session_dir(tmp_path)
    all_files = list(session.path.rglob("*"))
    files = [p for p in all_files if p.is_file()]
    before = _snapshot(files)

    DatasetIntegrityValidator(session, svo).validate()
    DatasetIntegrityValidator(session, svo).validate()  # twice, for determinism too

    after = _snapshot(files)
    assert before == after


def test_validation_is_deterministic(tmp_path):
    session, svo = _make_full_valid_session_dir(tmp_path)
    r1 = DatasetIntegrityValidator(session, svo).validate()
    r2 = DatasetIntegrityValidator(session, svo).validate()
    assert r1.summary_rows() == r2.summary_rows()
    assert r1.overall_status() == r2.overall_status()


# --------------------------------------------------------------------------- #
# 2-3. Manifest: missing / malformed.
# --------------------------------------------------------------------------- #

def test_missing_manifest_is_warning_not_error(tmp_path):
    root = tmp_path / "sess"
    root.mkdir()
    report = validate_manifest(root)
    assert report.status == Status.WARNING  # the app already defaults gracefully -- not a hard failure


def test_malformed_manifest_is_error_and_does_not_raise(tmp_path):
    root = tmp_path / "sess"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "dataset_manifest.json").write_text("{not valid json")

    report = validate_manifest(root)  # must not raise
    assert report.status == Status.ERROR


def test_manifest_that_is_a_json_array_not_object_is_error(tmp_path):
    root = tmp_path / "sess"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "dataset_manifest.json").write_text("[1, 2, 3]")

    report = validate_manifest(root)
    assert report.status == Status.ERROR


# --------------------------------------------------------------------------- #
# 4. Missing SVO2 / camera.
# --------------------------------------------------------------------------- #

def test_missing_svo2_is_warning(tmp_path):
    root = tmp_path / "sess"
    root.mkdir()
    report = validate_camera(root, svo=None)
    assert report.status == Status.WARNING
    assert any("missing" in c.detail for c in report.checks)


def test_svo2_present_but_not_opened_is_unavailable(tmp_path):
    root = tmp_path / "sess"
    _touch(root / "raw" / "recording.svo2")
    report = validate_camera(root, svo=None)
    assert report.status == Status.UNAVAILABLE


def test_svo_with_zero_frames_is_error(tmp_path):
    root = tmp_path / "sess"
    _touch(root / "raw" / "recording.svo2")
    svo = FakeSvo(np.array([], dtype=np.int64))
    report = validate_camera(root, svo)
    assert report.status == Status.ERROR


# --------------------------------------------------------------------------- #
# 5. Invalid/duplicate frame timestamps.
# --------------------------------------------------------------------------- #

def test_invalid_frame_timestamps_is_error():
    frames = pd.DataFrame({"timestamp_ns": np.array([0, np.nan, 200], dtype=np.float64)})
    report = validate_frames(frames)
    assert report.status == Status.ERROR


def test_duplicate_frame_timestamps_is_warning():
    frames = pd.DataFrame({"timestamp_ns": np.array([0, 100, 100, 200], dtype=np.int64)})
    report = validate_frames(frames)
    assert report.status == Status.WARNING
    dup_check = next(c for c in report.checks if c.name == "duplicate frame timestamps")
    assert "1" in dup_check.detail


def test_empty_frames_is_error():
    report = validate_frames(pd.DataFrame())
    assert report.status == Status.ERROR


# --------------------------------------------------------------------------- #
# 6. Radar: missing/unreadable heatmap and points files.
# --------------------------------------------------------------------------- #

def test_missing_radar_heatmap_file_is_error(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    session.radar = pd.DataFrame({
        "timestamp_ns": np.array([50], dtype=np.int64),
        "heatmap_path": ["radar/heatmap/does_not_exist.npy"],
        "points_path": [str((session.path / "radar" / "points" / "000.json").relative_to(session.path))],
    })
    report = validate_radar(session)
    assert report.status == Status.ERROR
    assert report.info["missing_heatmap_files"] == 1


def test_missing_radar_points_file_is_error(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    session.radar = pd.DataFrame({
        "timestamp_ns": np.array([50], dtype=np.int64),
        "heatmap_path": [str((session.path / "radar" / "heatmap" / "000.npy").relative_to(session.path))],
        "points_path": ["radar/points/does_not_exist.json"],
    })
    report = validate_radar(session)
    assert report.status == Status.ERROR
    assert report.info["missing_points_files"] == 1


def test_unreadable_radar_heatmap_file_is_error(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    bad_heatmap = session.path / "radar" / "heatmap" / "bad.npy"
    _touch(bad_heatmap, b"not a real npy file")
    session.radar = pd.DataFrame({
        "timestamp_ns": np.array([50], dtype=np.int64),
        "heatmap_path": [str(bad_heatmap.relative_to(session.path))],
        "points_path": [str((session.path / "radar" / "points" / "000.json").relative_to(session.path))],
    })
    report = validate_radar(session)
    assert report.status == Status.ERROR
    assert report.info["unreadable_heatmap_files"] == 1


def test_unreadable_radar_points_file_is_error(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    bad_points = session.path / "radar" / "points" / "bad.json"
    _touch(bad_points, b"{not valid json")
    session.radar = pd.DataFrame({
        "timestamp_ns": np.array([50], dtype=np.int64),
        "heatmap_path": [str((session.path / "radar" / "heatmap" / "000.npy").relative_to(session.path))],
        "points_path": [str(bad_points.relative_to(session.path))],
    })
    report = validate_radar(session)
    assert report.status == Status.ERROR
    assert report.info["unreadable_points_files"] == 1


def test_no_radar_metadata_is_warning_not_error():
    session = FakeSession(Path("/nonexistent"), radar=pd.DataFrame())
    report = validate_radar(session)
    assert report.status == Status.WARNING  # radar is an optional sensor


# --------------------------------------------------------------------------- #
# 7. Lidar: missing pcap/metadata, unavailable dependency.
# --------------------------------------------------------------------------- #

def test_missing_lidar_pcap_is_warning(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    (session.path / "raw" / "lidar.pcap").unlink()
    session.lidar = None
    report = validate_lidar(session)
    assert report.status != Status.ERROR
    pcap_check = next(c for c in report.checks if c.name == "lidar.pcap")
    assert pcap_check.status == Status.WARNING


def test_missing_lidar_metadata_json_is_warning(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    (session.path / "raw" / "lidar_0.json").unlink()
    session.lidar = None
    report = validate_lidar(session)
    meta_check = next(c for c in report.checks if c.name == "lidar_0.json")
    assert meta_check.status == Status.WARNING


def test_lidar_reader_open_error_is_reported_as_error(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    session.lidar = None
    session.lidar_error = "simulated parse failure"
    report = validate_lidar(session)
    assert report.status == Status.ERROR


def test_lidar_dependency_unavailable_is_reported(tmp_path, monkeypatch):
    session, _ = _make_full_valid_session_dir(tmp_path)
    session.lidar = None
    session.lidar_error = None  # files exist, but nothing loaded them

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ouster.sdk":
            raise ImportError("simulated: ouster-sdk not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    report = validate_lidar(session)
    assert report.status == Status.UNAVAILABLE


# --------------------------------------------------------------------------- #
# 8. MTi: missing HDF5, invalid timestamps.
# --------------------------------------------------------------------------- #

def test_missing_mti_hdf5_is_warning(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    session.mti = None
    report = validate_mti(session)
    assert report.status == Status.WARNING


def test_invalid_mti_timestamps_is_error(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    session.mti = FakeMti([0.0, float("nan"), 200.0])
    report = validate_mti(session)
    assert report.status == Status.ERROR


# --------------------------------------------------------------------------- #
# 9. GNSS: empty raw NMEA with valid MTi fused position.
# --------------------------------------------------------------------------- #

def test_empty_raw_gnss_with_valid_mti_fix_is_not_a_failure(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    (session.path / "raw" / "gnss.nmea.log").write_bytes(b"")  # empty
    session.mti = FakeMti(np.arange(0, 100, 10), latitude=np.full(10, 45.0), longitude=np.full(10, 9.0))

    report = validate_gnss(session)
    availability = next(c for c in report.checks if c.name == "GNSS viewer availability")
    assert availability.status == Status.PASS  # available via MTi fusion despite empty NMEA
    nmea_check = next(c for c in report.checks if c.name == "raw NMEA log")
    assert nmea_check.status == Status.WARNING  # noted, but not treated as invalidating the session


def test_empty_raw_gnss_without_mti_fix_is_unavailable_but_not_error(tmp_path):
    session, _ = _make_full_valid_session_dir(tmp_path)
    (session.path / "raw" / "gnss.nmea.log").write_bytes(b"")
    session.mti = FakeMti(np.arange(0, 100, 10))  # no valid lat/lon

    report = validate_gnss(session)
    assert report.status != Status.ERROR
    availability = next(c for c in report.checks if c.name == "GNSS viewer availability")
    assert availability.status == Status.WARNING


# --------------------------------------------------------------------------- #
# 10. Events / drop reporting.
# --------------------------------------------------------------------------- #

def test_event_and_drop_reporting_reflects_raw_metadata_without_interpretation():
    events = pd.DataFrame({
        "timestamp_ns": np.array([0, 10, 20], dtype=np.int64),
        "event_type": ["RECORDING_START", "RADAR_DROPPED_FRAMES", "RECORDING_STOP"],
        "description": ["session=x", "dropped=7", "session=x"],
    })
    report = validate_events(FakeSession(Path("/nonexistent"), events=events))
    assert report.info["event_count"] == 3
    assert report.info["earliest_event_ns"] == 0
    assert report.info["latest_event_ns"] == 20
    assert report.info["drop_events"] == [{"event_type": "RADAR_DROPPED_FRAMES", "description": "dropped=7"}]
    # The description string is surfaced verbatim ("dropped=7"), never
    # parsed into a claim about *why* frames were dropped.
    drop_check = next(c for c in report.checks if c.name == "sensor drop events")
    assert "dropped=7" in drop_check.detail
    assert report.status == Status.WARNING


def test_no_events_is_warning():
    report = validate_events(FakeSession(Path("/nonexistent"), events=pd.DataFrame()))
    assert report.status == Status.WARNING


def test_no_drop_events_is_pass():
    events = pd.DataFrame({
        "timestamp_ns": np.array([0], dtype=np.int64),
        "event_type": ["RECORDING_START"],
        "description": ["session=x"],
    })
    report = validate_events(FakeSession(Path("/nonexistent"), events=events))
    assert report.status == Status.PASS


# --------------------------------------------------------------------------- #
# 11. Timestamp coverage reporting.
# --------------------------------------------------------------------------- #

def test_timestamp_coverage_reports_ranges_relative_to_camera_start(tmp_path):
    session, svo = _make_full_valid_session_dir(tmp_path)
    report = validate_timestamp_coverage(session, svo)

    lo, hi = report.info["camera_range_s"]
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(400 / 1e9)  # frames span timestamps 0..400 ns

    # Sanity: every sensor's range is present, none are inverted (lo > hi).
    for key in ("camera_range_s", "radar_range_s", "lidar_range_s", "mti_range_s", "svo_range_s"):
        rng = report.info[key]
        assert rng is not None
        assert rng[0] <= rng[1]


def test_timestamp_coverage_marks_missing_sensor_unavailable_not_invented(tmp_path):
    session, svo = _make_full_valid_session_dir(tmp_path)
    session.lidar = None
    report = validate_timestamp_coverage(session, svo)
    assert report.info["lidar_range_s"] is None
    lidar_check = next(c for c in report.checks if c.name == "lidar")
    assert lidar_check.status == Status.UNAVAILABLE


# --------------------------------------------------------------------------- #
# 12-13. Status classification / structured result.
# --------------------------------------------------------------------------- #

def test_worst_status_ordering():
    assert worst_status(Status.PASS, Status.WARNING) == Status.WARNING
    assert worst_status(Status.WARNING, Status.ERROR) == Status.ERROR
    assert worst_status(Status.PASS, Status.UNAVAILABLE) == Status.UNAVAILABLE
    assert worst_status(Status.UNAVAILABLE, Status.WARNING) == Status.WARNING
    assert worst_status() == Status.PASS


def test_section_report_finalize_rolls_up_to_worst_check():
    report = SectionReport()
    report.add("a", Status.PASS)
    report.add("b", Status.WARNING)
    report.add("c", Status.PASS)
    assert report.finalize() == Status.WARNING


def test_validation_result_exposes_structured_sections_not_only_strings(tmp_path):
    session, svo = _make_full_valid_session_dir(tmp_path)
    result = DatasetIntegrityValidator(session, svo).validate()

    assert isinstance(result, ValidationResult)
    assert isinstance(result.radar, SectionReport)
    assert isinstance(result.radar.info, dict)
    assert isinstance(result.radar.info["row_count"], int)  # machine-readable, not just text
    names = [name for name, _ in result.all_sections()]
    assert "Radar" in names
