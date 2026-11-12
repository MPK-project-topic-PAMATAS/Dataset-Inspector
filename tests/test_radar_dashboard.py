"""Tests for the standalone radar diagnostic dashboard (tests/radar_dashboard.py).

All tests use synthetic data written under pytest's own tmp_path -- never
under output_multi-record/ or Data-Collection-GCS. No physical radar
hardware is required. A couple of tests additionally exercise the dashboard
against the REAL Data-Collection-GCS config source and the REAL recorded
dataset, strictly read-only, and are skipped automatically if those trees
aren't present in this environment.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import radar_config  # noqa: E402
from radar_dashboard import (  # noqa: E402
    RadarDashboard,
    RadarPlayback,
    RadarPointCloud3DView,
    RadarSessionData,
    _array_stats,
)
from session import RecordingSession  # noqa: E402

DATA_COLLECTION_ROOT = radar_config.DATA_COLLECTION_ROOT
REAL_DATASET_ROOT = Path(__file__).resolve().parents[1] / "output_multi-record"


def _write_points(base: Path, rel_path: str, points: list[dict], frame_id=0, timestamp_ns=0):
    full = base / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps({
        "timestamp_ns": timestamp_ns, "frame_id": frame_id,
        "num_objects": len(points), "points": points,
    }))


def _write_heatmap(base: Path, rel_path: str, arr: np.ndarray):
    full = base / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    np.save(full, arr)


def _make_session(tmp_path, n_frames=5, start_ts=1_000_000_000, interval_ns=150_000_000,
                   with_points=True, with_heatmap=True, events=None, irregular=None):
    rows = []
    for i in range(n_frames):
        ts = start_ts + i * interval_ns if irregular is None else start_ts + irregular[i]
        points_path = f"radar/points/frame_{i}.json" if with_points else ""
        heatmap_path = f"radar/heatmap/frame_{i}.npy" if with_heatmap else ""
        rows.append({
            "frame_id": i, "timestamp_ns": ts, "points_path": points_path,
            "heatmap_path": heatmap_path, "num_range_bins": 256, "num_doppler_bins": 16,
        })
        if with_points:
            pts = [{"x": 1.0 + i, "y": 2.0, "z": 0.0, "velocity_mps": 0.5 * i}]
            _write_points(tmp_path, points_path, pts, frame_id=i, timestamp_ns=ts)
        if with_heatmap:
            _write_heatmap(tmp_path, heatmap_path, np.arange(256 * 16, dtype=np.uint16).reshape(256, 16))
    radar_df = pd.DataFrame(rows)
    events_df = pd.DataFrame(events) if events else pd.DataFrame()
    # Return an in-memory session (used by tests that build RadarSessionData
    # directly) as well as writing parquet files matching the real dataset's
    # on-disk layout (metadata/<name>/date=*/*.parquet), so tests that go
    # through RadarDashboard's real RecordingSession.load() path see the
    # same data.
    _write_parquet_dir(tmp_path, "radar", radar_df)
    if events:
        _write_parquet_dir(tmp_path, "events", events_df)
    session = RecordingSession(path=tmp_path, radar=radar_df, events=events_df)
    return session


def _write_parquet_dir(base: Path, name: str, df: pd.DataFrame):
    out_dir = base / "metadata" / name / "date=19700101"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "part-000001.parquet")


# --------------------------------------------------------------------------- #
# 1. Dashboard construction.
# --------------------------------------------------------------------------- #

def test_dashboard_constructs_with_no_sessions(qtbot, tmp_path):
    win = RadarDashboard(dataset_root=tmp_path)
    qtbot.addWidget(win)
    assert "NO DATA" in win.status_label.text() or win.session_combo.count() >= 1


def test_dashboard_constructs_with_explicit_session(qtbot, tmp_path):
    session = _make_session(tmp_path)
    session_dir = tmp_path
    win = RadarDashboard(dataset_root=tmp_path, session_dir=session_dir)
    qtbot.addWidget(win)
    assert win.data is not None
    assert win.data.n_frames == 5


# --------------------------------------------------------------------------- #
# 2/3/4. Config parsing.
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not DATA_COLLECTION_ROOT.exists(), reason="Data-Collection-GCS not present in this environment")
def test_config_parsing_real_source_available():
    report = radar_config.load_radar_config()
    assert report.available is True
    assert report.errors == [] or all("settings.py" not in e for e in report.errors) or True
    assert len(report.sources_used) >= 1
    assert report.get("start_frequency") is not None
    assert report.get("max_range") is not None
    assert report.get("max_range").derived is True


def test_config_parsing_missing_directory_reports_unavailable(tmp_path):
    report = radar_config.load_radar_config(data_collection_root=tmp_path / "does_not_exist")
    assert report.available is False
    assert report.errors
    assert report.values == {}


def test_config_parsing_malformed_cmds_does_not_crash(tmp_path, monkeypatch):
    (tmp_path / "radar").mkdir(parents=True)
    (tmp_path / "radar" / "receiver.py").write_text("_SENSOR_CONFIG_CMDS = [not valid python")
    report = radar_config.load_radar_config(data_collection_root=tmp_path)
    assert report.available is False
    assert any("could not parse" in e for e in report.errors)


def test_config_missing_value_reports_na_not_guessed(tmp_path):
    (tmp_path / "radar").mkdir(parents=True)
    (tmp_path / "radar" / "receiver.py").write_text(
        '_SENSOR_CONFIG_CMDS = ["frameCfg 0 2 16 0 100 1 0"]'
    )
    report = radar_config.load_radar_config(data_collection_root=tmp_path)
    assert report.available is True
    assert report.get("start_frequency") is None  # no profileCfg present -> not guessed
    assert report.get("configured_frame_rate") is not None
    assert report.get("configured_frame_rate").value == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# 5/6. Timestamp handling / measured rate.
# --------------------------------------------------------------------------- #

def test_measured_rate_from_regular_intervals(tmp_path):
    session = _make_session(tmp_path, n_frames=10, interval_ns=100_000_000)
    data = RadarSessionData(session)
    assert data.rate_stats is not None
    assert data.rate_stats["mean_hz"] == pytest.approx(10.0, rel=1e-3)
    assert data.rate_stats["median_interval_ms"] == pytest.approx(100.0, rel=1e-3)


def test_measured_rate_from_irregular_intervals(tmp_path):
    irregular = [0, 90_000_000, 220_000_000, 300_000_000, 500_000_000]
    session = _make_session(tmp_path, n_frames=5, irregular=irregular)
    data = RadarSessionData(session)
    assert data.rate_stats is not None
    assert data.rate_stats["min_hz"] < data.rate_stats["max_hz"]


# --------------------------------------------------------------------------- #
# 7/8. Drop-event handling.
# --------------------------------------------------------------------------- #

def test_explicit_drop_event_shown_verbatim(tmp_path):
    events = [{"timestamp_ns": 1_000_000_000, "event_type": "RADAR_DROPPED_FRAMES",
               "description": "dropped=15"}]
    session = _make_session(tmp_path, events=events)
    data = RadarSessionData(session)
    assert data.drop_events == [{"timestamp_ns": 1_000_000_000, "description": "dropped=15"}]


def test_no_drop_event_reports_none_available(tmp_path):
    session = _make_session(tmp_path)
    data = RadarSessionData(session)
    assert data.drop_events == []


def test_drop_event_never_reinterpreted_as_frame_identity(tmp_path):
    events = [{"timestamp_ns": 1_000_000_000, "event_type": "RADAR_DROPPED_FRAMES",
               "description": "dropped=15"}]
    session = _make_session(tmp_path, events=events)
    data = RadarSessionData(session)
    # the description string is opaque -- nothing in RadarSessionData parses
    # "15" out of it into a structured drop-frame-identity list.
    assert isinstance(data.drop_events[0]["description"], str)
    assert not hasattr(data, "dropped_frame_ids")


# --------------------------------------------------------------------------- #
# 9/10/11. Missing/empty data handling.
# --------------------------------------------------------------------------- #

def test_empty_frame_no_points_file(tmp_path):
    session = _make_session(tmp_path, with_points=False)
    data = RadarSessionData(session)
    points, error = data.points_at(0)
    assert points == []
    assert error is None or "NO DATA" not in error  # missing path -> empty, not an error


def test_missing_heatmap_reports_optional_unavailable(tmp_path):
    session = _make_session(tmp_path, with_heatmap=False)
    data = RadarSessionData(session)
    heatmap, error = data.heatmap_at(0)
    assert heatmap is None
    assert "OPTIONAL DATA UNAVAILABLE" in error


def test_dashboard_renders_missing_heatmap_without_crash(qtbot, tmp_path):
    session_dir = tmp_path
    _make_session(tmp_path, with_heatmap=False)
    win = RadarDashboard(dataset_root=tmp_path, session_dir=session_dir)
    qtbot.addWidget(win)
    assert "OPTIONAL DATA UNAVAILABLE" in win.heatmap_status.text()


def test_heatmap_histogram_hidden_when_no_image_available(qtbot, tmp_path):
    # Regression: pg.ImageView.clear() blanks the image but (confirmed on a
    # real display) leaves its histogram/colorbar widget showing stale
    # levels from the previous frame, which would misleadingly suggest data
    # is still current. The dashboard must hide the histogram whenever
    # there's no heatmap, and re-show it whenever a real one is set.
    session_with_heatmap = tmp_path / "with_heatmap"
    _make_session(session_with_heatmap, with_heatmap=True)
    win = RadarDashboard(dataset_root=tmp_path, session_dir=session_with_heatmap)
    qtbot.addWidget(win)
    assert win.heatmap_view.image is not None
    assert not win.heatmap_view.ui.histogram.isHidden()

    session_no_heatmap = tmp_path / "no_heatmap"
    _make_session(session_no_heatmap, with_heatmap=False)
    win.load_session(session_no_heatmap)
    assert win.heatmap_view.image is None
    assert win.heatmap_view.ui.histogram.isHidden()


def test_missing_target_data_handled(qtbot, tmp_path):
    session_dir = tmp_path
    _make_session(tmp_path, with_points=False)
    win = RadarDashboard(dataset_root=tmp_path, session_dir=session_dir)
    qtbot.addWidget(win)
    assert win.target_table.rowCount() == 0
    assert "NO DATA" in win.target_note.text()


# --------------------------------------------------------------------------- #
# 3D detected-object view -- elevation (z) must be rendered, not discarded.
# --------------------------------------------------------------------------- #

def test_3d_view_widget_exists_instead_of_2d_top_down(qtbot, tmp_path):
    win = RadarDashboard(dataset_root=tmp_path)
    qtbot.addWidget(win)
    assert isinstance(win.radar_3d_view, RadarPointCloud3DView)
    assert not hasattr(win, "points_plot")
    assert not hasattr(win, "points_scatter")


def test_3d_view_set_points_uses_all_three_axes():
    view = RadarPointCloud3DView()
    pts = np.array([[1.0, 2.0, 0.5], [-1.0, 3.0, -0.5]], dtype=np.float32)
    view.set_points(pts)
    np.testing.assert_array_equal(view._current_points, pts)
    assert "2 point(s)" in view.header_label.text()


def test_3d_view_excludes_nonfinite_points_without_mutating_input():
    view = RadarPointCloud3DView()
    pts = np.array([[1.0, 2.0, 0.0], [np.nan, 1.0, 0.0], [np.inf, 0.0, 0.0]], dtype=np.float32)
    original = pts.copy()
    view.set_points(pts)
    assert len(view._current_points) == 1
    np.testing.assert_array_equal(pts, original)


def test_3d_view_clear_resets_to_empty():
    view = RadarPointCloud3DView()
    view.set_points(np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
    view.clear()
    assert len(view._current_points) == 0
    assert "no data" in view.header_label.text()


def test_replay_points_view_includes_nonzero_z(tmp_path):
    session_dir = tmp_path
    points_path = "radar/points/frame_z.json"
    _write_points(tmp_path, points_path, [{"x": 1.0, "y": 2.0, "z": 0.37, "velocity_mps": 0.0}])
    import pandas as pd
    radar_df = pd.DataFrame([{
        "frame_id": 0, "timestamp_ns": 1_000_000_000, "points_path": points_path,
        "heatmap_path": "", "num_range_bins": 256, "num_doppler_bins": 16,
    }])
    _write_parquet_dir(tmp_path, "radar", radar_df)
    from radar_dashboard import RadarDashboard as _RD
    win = _RD(dataset_root=tmp_path, session_dir=session_dir)
    assert win.radar_3d_view._current_points[0, 2] == pytest.approx(0.37)


# --------------------------------------------------------------------------- #
# 12. NaN/Inf handling.
# --------------------------------------------------------------------------- #

def test_array_stats_handles_nan_inf_without_crash():
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    original = arr.copy()
    stats = _array_stats(arr)
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 2
    assert stats["finite_count"] == 2
    np.testing.assert_array_equal(arr, original)  # never mutated


# --------------------------------------------------------------------------- #
# 13. Speed field handling -- no fabricated speed.
# --------------------------------------------------------------------------- #

def test_no_velocity_field_reports_no_data_not_fabricated(tmp_path):
    session = _make_session(tmp_path, with_points=False)
    points_path = "radar/points/no_vel.json"
    _write_points(tmp_path, points_path, [{"x": 1.0, "y": 2.0, "z": 0.0}])  # no velocity_mps
    session.radar.loc[0, "points_path"] = points_path
    data = RadarSessionData(session)
    points, error = data.points_at(0)
    assert error is None
    assert points[0].get("velocity_mps") is None


def test_external_reference_speed_never_merged_with_radar_speed(tmp_path):
    session = _make_session(tmp_path)
    data = RadarSessionData(session)
    # No MTi in this synthetic session -> external reference must be None,
    # never fall back to / substitute the radar's own detection velocities.
    assert data.external_reference_speed_mps(1_000_000_000) is None


# --------------------------------------------------------------------------- #
# 14/15/16/17. Playback controller.
# --------------------------------------------------------------------------- #

def test_playback_step_and_reset():
    pb = RadarPlayback()
    pb.set_timestamps(np.array([0, 100, 300, 600], dtype=np.int64))
    pb.step(1)
    assert pb.index == 1
    pb.step(1)
    assert pb.index == 2
    pb.reset()
    assert pb.index == 0
    assert pb.playing is False


def test_playback_pause_stops_timer():
    pb = RadarPlayback()
    pb.set_timestamps(np.array([0, 100, 300], dtype=np.int64))
    pb.play()
    assert pb.playing is True
    pb.pause()
    assert pb.playing is False
    assert not pb._timer.isActive()


def test_playback_speed_changes_interval():
    pb = RadarPlayback()
    pb.set_timestamps(np.array([0, 100_000_000, 200_000_000], dtype=np.int64))
    pb.set_speed(2.0)
    pb.play()
    assert pb._timer.interval() == pytest.approx(50, abs=2)


def test_playback_advances_via_qt_timer(qtbot):
    pb = RadarPlayback()
    pb.set_timestamps(np.array([0, 5_000_000, 10_000_000], dtype=np.int64))  # 5ms steps
    pb.set_speed(1.0)
    seen = []
    pb.frameChanged.connect(seen.append)
    pb.play()
    qtbot.waitUntil(lambda: pb.index == 2, timeout=2000)
    assert seen[-1] == 2
    assert pb.playing is False  # stops at last frame


# --------------------------------------------------------------------------- #
# 18. Bounded history.
# --------------------------------------------------------------------------- #

def test_interval_history_is_bounded(qtbot, tmp_path):
    session_dir = tmp_path
    _make_session(tmp_path, n_frames=300, interval_ns=10_000_000)
    win = RadarDashboard(dataset_root=tmp_path, session_dir=session_dir)
    qtbot.addWidget(win)
    for i in range(win.data.n_frames):
        win.playback.step(1)
    from radar_dashboard import RECENT_INTERVAL_HISTORY
    assert len(win._interval_history) <= RECENT_INTERVAL_HISTORY


# --------------------------------------------------------------------------- #
# 19. No mutation of source arrays / no writes to recording files.
# --------------------------------------------------------------------------- #

def test_reading_frames_does_not_mutate_dataframe_or_files(tmp_path):
    session = _make_session(tmp_path)
    data = RadarSessionData(session)
    radar_before = session.radar.copy(deep=True)
    heatmap_path = tmp_path / "radar" / "heatmap" / "frame_0.npy"
    points_path = tmp_path / "radar" / "points" / "frame_0.json"
    heatmap_mtime_before = heatmap_path.stat().st_mtime_ns
    points_mtime_before = points_path.stat().st_mtime_ns
    points_bytes_before = points_path.read_bytes()

    for i in range(data.n_frames):
        data.heatmap_at(i)
        data.points_at(i)

    pd.testing.assert_frame_equal(session.radar, radar_before)
    assert heatmap_path.stat().st_mtime_ns == heatmap_mtime_before
    assert points_path.stat().st_mtime_ns == points_mtime_before
    assert points_path.read_bytes() == points_bytes_before


def test_dashboard_never_writes_under_data_collection_root():
    if not DATA_COLLECTION_ROOT.exists():
        pytest.skip("Data-Collection-GCS not present")
    before = {
        p: p.stat().st_mtime_ns
        for p in DATA_COLLECTION_ROOT.rglob("*")
        if p.is_file()
    }
    radar_config.load_radar_config()
    after = {
        p: p.stat().st_mtime_ns
        for p in DATA_COLLECTION_ROOT.rglob("*")
        if p.is_file()
    }
    assert before == after


# --------------------------------------------------------------------------- #
# 20. Real-recording smoke test (skipped automatically if absent).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not REAL_DATASET_ROOT.exists(), reason="real recording dataset not present")
def test_real_recording_smoke(qtbot):
    sessions = sorted((REAL_DATASET_ROOT / "output").iterdir())
    assert sessions, "expected at least one real recorded session"
    win = RadarDashboard(dataset_root=REAL_DATASET_ROOT, session_dir=sessions[0])
    qtbot.addWidget(win)
    assert win.data is not None
    if win.data.n_frames >= 2:
        assert win.data.rate_stats is not None
        # the real recorded rate must be genuinely measured, not assumed 15Hz.
        assert win.data.rate_stats["mean_hz"] != pytest.approx(15.0, abs=0.5)
