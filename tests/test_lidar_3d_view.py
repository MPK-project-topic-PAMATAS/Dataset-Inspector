"""Tests for the interactive 3D LiDAR point-cloud viewer (lidar_3d_view.py)
and its wiring into MainWindow.

Lidar3DView is a pure presentation widget -- these tests build small real
numpy point arrays and sensor_health.SensorDiagnostic instances rather than
touching LidarReader/RecordingSession, to prove the widget only consumes
already-selected data. Real OpenGL pixel output cannot be asserted on a
headless CI platform (QT_QPA_PLATFORM=offscreen cannot actually paint a
QOpenGLWidget -- verified manually against a real X display instead, see
the session report), so these tests exercise construction, state, and the
data/camera math instead of rendered pixels.
"""
from __future__ import annotations

import builtins
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensor_health import SensorDiagnostic, SensorHealthModel, SensorState
from session_loader import SessionLoadResult
from sync_index import MISSING


@pytest.fixture
def view(qtbot):
    from lidar_3d_view import Lidar3DView

    w = Lidar3DView()
    qtbot.addWidget(w)
    return w


# --------------------------------------------------------------------------- #
# Construction / empty / degenerate clouds.
# --------------------------------------------------------------------------- #

def test_widget_constructs_without_error(view):
    assert view.gl_view is not None
    assert view.scatter is not None
    assert "no session" in view.header_label.text().lower()


def test_missing_lidar_shows_explicit_message_not_a_fabricated_cloud(view):
    view.show_missing("No LiDAR data for this frame")
    assert "No LiDAR data for this frame" in view.header_label.text()
    assert len(view._current_points) == 0


def test_set_points_none_is_treated_as_missing(view):
    view.set_points(None)
    assert "No LiDAR data for this frame" in view.header_label.text()


def test_empty_point_array_renders_without_raising(view):
    view.set_points(np.zeros((0, 3), dtype=np.float32), scan_index=0, timestamp_ns=0)
    assert len(view._current_points) == 0


def test_one_point_cloud_renders_and_fits_without_division_by_zero(view):
    view.set_points(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), scan_index=1, timestamp_ns=1_000_000_000)
    assert len(view._current_points) == 1
    distance, center = view._fit_parameters()
    assert distance > 0
    assert center.x() == pytest.approx(1.0)


def test_small_cloud_renders(view):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    view.set_points(pts, scan_index=2, timestamp_ns=2_000_000_000)
    assert len(view._current_points) == 3


def test_large_synthetic_cloud_renders_without_raising(view):
    rng = np.random.default_rng(0)
    pts = (rng.random((150_000, 3), dtype=np.float32) - 0.5) * 40
    view.set_points(pts, scan_index=3, timestamp_ns=3_000_000_000)
    assert len(view._current_points) == 150_000


def test_degenerate_zero_extent_cloud_uses_minimum_extent_safely(view):
    pts = np.full((10, 3), 5.0, dtype=np.float32)  # every point identical -> zero extent
    view.set_points(pts, scan_index=4, timestamp_ns=4_000_000_000)
    distance, _center = view._fit_parameters()
    assert distance > 0 and np.isfinite(distance)


# --------------------------------------------------------------------------- #
# NaN/Inf rendering guard: presentation-only exclusion, never mutates input.
# --------------------------------------------------------------------------- #

def test_nan_inf_points_are_excluded_from_display_only(view):
    pts = np.array([
        [1.0, 1.0, 1.0],
        [np.nan, 0.0, 0.0],
        [0.0, np.inf, 0.0],
        [2.0, 2.0, 2.0],
    ], dtype=np.float32)
    original = pts.copy()

    view.set_points(pts, scan_index=5, timestamp_ns=5_000_000_000)

    assert len(view._current_points) == 2  # only the two finite rows displayed
    assert np.array_equal(pts, original, equal_nan=True)  # the array MainWindow handed in is untouched
    assert "non-finite excluded from rendering" in view.header_label.text()


def test_all_nan_cloud_renders_as_empty_without_raising(view):
    pts = np.full((5, 3), np.nan, dtype=np.float32)
    view.set_points(pts, scan_index=6, timestamp_ns=6_000_000_000)
    assert len(view._current_points) == 0


# --------------------------------------------------------------------------- #
# Header metadata: point count, scan index/timestamp, delta from SensorDiagnostic.
# --------------------------------------------------------------------------- #

def test_point_count_is_displayed(view):
    pts = np.zeros((1234, 3), dtype=np.float32)
    view.set_points(pts, scan_index=7, timestamp_ns=7_000_000_000)
    assert "1,234 points" in view.header_label.text()


def test_selected_scan_metadata_is_displayed(view):
    pts = np.zeros((10, 3), dtype=np.float32)
    view.set_points(pts, scan_index=412, timestamp_ns=8_123_456_789)
    text = view.header_label.text()
    assert "412" in text
    assert "8.123457" in text  # timestamp_ns / 1e9 formatted


def test_delta_display_comes_verbatim_from_existing_sensor_diagnostic(view):
    diag = SensorDiagnostic(
        name="Lidar", state=SensorState.AVAILABLE, sample_index=412,
        sensor_timestamp_ns=8_123_456_789, delta_ns=8_400_000, sync_class="GOOD",
    )
    pts = np.zeros((10, 3), dtype=np.float32)
    view.set_points(pts, scan_index=412, timestamp_ns=8_123_456_789, diagnostic=diag)

    text = view.header_label.text()
    assert "+8.4 ms" in text
    assert "AVAILABLE" in text
    # The widget must not have recomputed this value -- it's the exact
    # number already carried by the diagnostic object passed in.
    assert diag.delta_ms == pytest.approx(8.4)


def test_missing_diagnostic_state_is_shown_without_a_delta(view):
    diag = SensorDiagnostic(name="Lidar", state=SensorState.NO_DATA)
    pts = np.zeros((5, 3), dtype=np.float32)
    view.set_points(pts, scan_index=1, timestamp_ns=1_000_000_000, diagnostic=diag)
    assert "NO DATA" in view.header_label.text()


# --------------------------------------------------------------------------- #
# Camera controls: reset / fit-to-cloud only ever change the camera.
# --------------------------------------------------------------------------- #

def test_reset_view_does_not_alter_point_data(view):
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    view.set_points(pts, scan_index=1, timestamp_ns=1_000_000_000)
    before = view._current_points.copy()

    view.reset_view()

    assert np.array_equal(view._current_points, before)


def test_fit_to_cloud_does_not_alter_point_data(view):
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    view.set_points(pts, scan_index=1, timestamp_ns=1_000_000_000)
    before = view._current_points.copy()

    view.fit_to_cloud()

    assert np.array_equal(view._current_points, before)


def test_fit_to_cloud_on_empty_cloud_does_nothing_unsafe(view):
    view.show_missing("No LiDAR data for this frame")
    view.fit_to_cloud()  # must not raise, no division by zero
    view.reset_view()


def test_point_size_and_color_mode_changes_only_affect_rendering(view):
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    view.set_points(pts, scan_index=1, timestamp_ns=1_000_000_000)
    before = view._current_points.copy()

    view.point_size_slider.setValue(7)
    view.color_mode_combo.setCurrentText("Uniform")
    view.color_mode_combo.setCurrentText("Height")

    assert np.array_equal(view._current_points, before)
    assert view._point_size == 7.0


# --------------------------------------------------------------------------- #
# The widget/scatter/gl_view are created once and reused, never recreated.
# --------------------------------------------------------------------------- #

def test_repeated_updates_do_not_recreate_the_widget(view):
    gl_view_id = id(view.gl_view)
    scatter_id = id(view.scatter)

    for i in range(50):
        pts = np.random.default_rng(i).random((100, 3)).astype(np.float32)
        view.set_points(pts, scan_index=i, timestamp_ns=i * 1_000_000)

    assert id(view.gl_view) == gl_view_id
    assert id(view.scatter) == scatter_id


def test_resize_does_not_raise(view):
    pts = np.zeros((100, 3), dtype=np.float32)
    view.set_points(pts, scan_index=1, timestamp_ns=1_000_000_000)
    for w, h in [(800, 600), (50, 50), (1, 1), (2000, 200)]:
        view.resize(w, h)


def test_clear_resets_cloud_and_one_time_fit_state(view):
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    view.set_points(pts, scan_index=1, timestamp_ns=1_000_000_000)
    assert view._has_shown_any_cloud is True

    view.clear()

    assert len(view._current_points) == 0
    assert view._has_shown_any_cloud is False
    assert "no session" in view.header_label.text().lower()


# --------------------------------------------------------------------------- #
# No disk/PCAP access from any interaction handler.
# --------------------------------------------------------------------------- #

def test_no_disk_access_during_rotate_zoom_pan_reset_or_control_changes(view, monkeypatch):
    pts = np.random.default_rng(1).random((5000, 3)).astype(np.float32)
    view.set_points(pts, scan_index=1, timestamp_ns=1_000_000_000)

    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("Lidar3DView interaction must not touch disk")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        view.reset_view()
        view.fit_to_cloud()
        view.point_size_slider.setValue(5)
        view.color_mode_combo.setCurrentText("Uniform")
        view.gl_view.orbit(10, 5)          # "left-drag" equivalent
        view.gl_view.pan(1, 1, 0)          # "pan" equivalent
        view.gl_view.opts["distance"] *= 0.9  # "wheel zoom" equivalent
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


def test_widget_module_never_imports_pcap_or_session_machinery():
    import ast

    import lidar_3d_view as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)

    forbidden_modules = {"ouster", "ouster.sdk", "lidar_reader", "session", "sync_index"}
    hits = imported_names & forbidden_modules
    assert hits == set(), f"lidar_3d_view.py must stay a pure data-in widget, found imports: {hits}"


# --------------------------------------------------------------------------- #
# MainWindow integration: session replacement, stale-result protection.
# --------------------------------------------------------------------------- #

class FakeLidar:
    def __init__(self, timestamps_ns, points):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self.points = points

    @property
    def num_scans(self):
        return len(self.timestamps_ns)

    def points_at(self, index):
        return self.points[index]


class FakeSyncIndexWithLidar:
    def __init__(self, lidar_index=0):
        self._lidar_index = lidar_index

    def radar_for(self, i):
        return MISSING

    def lidar_for(self, i):
        return self._lidar_index

    def mti_for(self, i):
        return MISSING

    def svo_for(self, i):
        return MISSING


class FakeSession:
    def __init__(self, session_id, lidar=None):
        self.session_id = session_id
        self.path = Path("/nonexistent") / session_id
        self.frames = pd.DataFrame({"timestamp_ns": np.arange(5, dtype=np.int64) * 1_000_000})
        self.radar = pd.DataFrame()
        self.events = pd.DataFrame()
        self.mti = None
        self.lidar = lidar
        self.lidar_error = None
        self.manifest = {}
        self.session_manifest = {}

    @property
    def svo_path(self):
        return self.path / "raw" / "recording.svo2"


class FakeSvo:
    def __init__(self):
        self.closed = False
        self.timestamps_ns = np.array([0], dtype=np.int64)
        self.has_zed_imu = False

    def close(self):
        self.closed = True


@pytest.fixture
def window(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)
    yield win
    win._worker_busy = False
    win._pending_request = None
    win.close()


def _make_result(generation, session_id, lidar_points):
    lidar = FakeLidar([1_000_000, 2_000_000, 3_000_000], lidar_points)
    session = FakeSession(session_id, lidar=lidar)
    sync_index = FakeSyncIndexWithLidar(lidar_index=0)
    return SessionLoadResult(
        generation=generation, session_dir=session.path, ok=True,
        session=session, svo=FakeSvo(), sync_index=sync_index,
        validation_result=None, sensor_health=SensorHealthModel(session, sync_index),
    )


def test_session_commit_feeds_the_new_point_cloud_to_the_viewer(window):
    points = [np.array([[1, 1, 1]], dtype=np.float32)] * 3
    window._session_generation = 1
    window._on_session_load_finished(_make_result(1, "A", points))

    assert len(window.lidar_view._current_points) == 1
    assert np.array_equal(window.lidar_view._current_points[0], [1, 1, 1])


def test_session_replacement_does_not_retain_old_cloud(window):
    points_a = [np.array([[9, 9, 9]], dtype=np.float32)] * 3
    window._session_generation = 1
    window._on_session_load_finished(_make_result(1, "A", points_a))
    assert np.array_equal(window.lidar_view._current_points[0], [9, 9, 9])

    points_b = [np.array([[2, 2, 2]], dtype=np.float32)] * 3
    window._session_generation = 2
    window._on_session_load_finished(_make_result(2, "B", points_b))

    assert np.array_equal(window.lidar_view._current_points[0], [2, 2, 2])
    assert not np.array_equal(window.lidar_view._current_points[0], [9, 9, 9])


def test_stale_session_result_cannot_update_the_viewer(window):
    points_a = [np.array([[9, 9, 9]], dtype=np.float32)] * 3
    window._session_generation = 1
    window._on_session_load_finished(_make_result(1, "A", points_a))
    before = window.lidar_view._current_points.copy()

    window._session_generation = 2  # a newer request has been made
    stale = _make_result(1, "A-late-duplicate", [np.array([[7, 7, 7]], dtype=np.float32)] * 3)
    window._on_session_load_finished(stale)  # generation 1 is now obsolete

    assert np.array_equal(window.lidar_view._current_points, before)


def test_missing_lidar_session_shows_missing_state_in_viewer(window):
    window._session_generation = 1
    session = FakeSession("A", lidar=None)
    sync_index = FakeSyncIndexWithLidar()
    result = SessionLoadResult(
        generation=1, session_dir=session.path, ok=True, session=session, svo=None,
        sync_index=sync_index, validation_result=None, sensor_health=SensorHealthModel(session, sync_index),
    )
    window._on_session_load_finished(result)

    assert len(window.lidar_view._current_points) == 0
    assert "no lidar" in window.lidar_view.header_label.text().lower()
