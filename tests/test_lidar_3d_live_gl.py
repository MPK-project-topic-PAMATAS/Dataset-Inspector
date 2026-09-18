"""Live-OpenGL performance tests for the Ouster 3D LiDAR viewer.

These measure *actual GPU-presented frame cadence* (paintGL invocations),
which is fundamentally different from Qt event-loop iteration count or
data-path latency (see test_lidar_3d_performance.py for those) --
QT_QPA_PLATFORM=offscreen cannot paint a QOpenGLWidget at all, so these
tests are skipped by default and only run when explicitly opted into with
a real, working GL-capable display:

    DISPLAY=:1 LIDAR3D_LIVE_GL=1 pytest tests/test_lidar_3d_live_gl.py

They must never run as part of the default headless suite, must never be
required for the default 241+ baseline to pass, and must never make the
suite depend on a specific GPU/driver -- they verify against the FPS/
frame-interval budgets from the design brief, not exact hardware-specific
numbers.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest


def _live_gl_requested() -> bool:
    return os.environ.get("LIDAR3D_LIVE_GL") == "1" and bool(os.environ.get("DISPLAY"))


def _live_gl_usable() -> bool:
    """Actually try to paint one frame and confirm a real GPU surface
    exists, rather than trusting environment variables alone."""
    if not _live_gl_requested():
        return False
    try:
        import sys

        from PySide6.QtWidgets import QApplication
        from lidar_3d_view import Lidar3DView

        app = QApplication.instance() or QApplication(sys.argv)
        v = Lidar3DView()
        v.resize(200, 200)
        v.show()
        v.set_points(np.zeros((10, 3), dtype=np.float32), scan_index=0, timestamp_ns=0)
        for _ in range(10):
            app.processEvents()
        pix = v.grab()
        v.close()
        return not pix.isNull()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _live_gl_usable(),
    reason=(
        "live OpenGL not available/requested -- set LIDAR3D_LIVE_GL=1 and "
        "DISPLAY to a real, GL-capable X server to run these (they never "
        "run under QT_QPA_PLATFORM=offscreen, which cannot paint a "
        "QOpenGLWidget)"
    ),
)


def _synthetic_cloud(n, seed=0):
    rng = np.random.default_rng(seed)
    return ((rng.random((n, 3), dtype=np.float32) - 0.5) * 20.0)


def _run_for(app, duration_s, action):
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        action()
        app.processEvents()


def _capture_paints(view, app, duration_s, action):
    paint_times = []
    orig_paint = view.gl_view.paintGL

    def counted():
        paint_times.append(time.perf_counter())
        orig_paint()

    view.gl_view.paintGL = counted
    try:
        _run_for(app, duration_s, action)
    finally:
        view.gl_view.paintGL = orig_paint
    return paint_times


def _fps_and_intervals(paint_times):
    if len(paint_times) < 2:
        return 0.0, np.array([0.0]), 0.0
    intervals = np.diff(paint_times) * 1000
    fps = (len(paint_times) - 1) / (paint_times[-1] - paint_times[0])
    return fps, intervals, intervals.max()


@pytest.fixture
def live_view(qtbot):
    from PySide6.QtWidgets import QApplication
    from lidar_3d_view import Lidar3DView

    app = QApplication.instance()
    v = Lidar3DView()
    v.resize(800, 600)
    v.show()
    v.set_points(_synthetic_cloud(4000, seed=1), scan_index=0, timestamp_ns=0)
    for _ in range(10):
        app.processEvents()
    yield v, app
    v.close()


# --------------------------------------------------------------------------- #
# 1C / 3: rotation frame cadence.
# --------------------------------------------------------------------------- #

def test_rotation_meets_fps_budget(live_view):
    view, app = live_view
    paints = _capture_paints(view, app, 2.0, lambda: view.gl_view.orbit(3, 0))
    fps, intervals, worst = _fps_and_intervals(paints)
    p95 = float(np.percentile(intervals, 95))
    print(f"\n[rotation] {len(paints)} paints, {fps:.1f} FPS, p95={p95:.1f}ms, worst={worst:.1f}ms")

    assert fps >= 20.0, f"HARD FAILURE: rotation FPS {fps:.1f} < 20"
    assert fps >= 24.0, f"acceptance budget violated: rotation FPS {fps:.1f} < 24"
    assert p95 <= 100.0, f"HARD FAILURE: frame interval p95 {p95:.1f}ms > 100ms"
    assert p95 <= 67.0, f"acceptance budget violated: frame interval p95 {p95:.1f}ms > 67ms"


def test_zoom_and_pan_do_not_stall(live_view):
    view, app = live_view

    zoom_paints = _capture_paints(
        view, app, 1.0,
        lambda: (view.gl_view.opts.__setitem__("distance", view.gl_view.opts["distance"] * 0.995), view.gl_view.update()),
    )
    pan_paints = _capture_paints(view, app, 1.0, lambda: view.gl_view.pan(1, 1, 0, relative="view-upright"))

    zoom_fps, zoom_intervals, _ = _fps_and_intervals(zoom_paints)
    pan_fps, pan_intervals, _ = _fps_and_intervals(pan_paints)
    print(f"\n[zoom] {zoom_fps:.1f} FPS  [pan] {pan_fps:.1f} FPS")

    assert zoom_fps >= 20.0
    assert pan_fps >= 20.0


# --------------------------------------------------------------------------- #
# Repeated cloud replacement (the LiDAR-viewer side of "playback") -- kept
# separate from MainWindow/camera/SVO decode per the design brief's
# instruction to measure the LiDAR path independently of overall playback.
# --------------------------------------------------------------------------- #

def test_repeated_cloud_replacement_meets_fps_budget(live_view):
    view, app = live_view
    counter = {"i": 0}

    def replace():
        counter["i"] += 1
        view.set_points(_synthetic_cloud(4000, seed=counter["i"]), scan_index=counter["i"], timestamp_ns=counter["i"])

    paints = _capture_paints(view, app, 2.0, replace)
    fps, intervals, worst = _fps_and_intervals(paints)
    p95 = float(np.percentile(intervals, 95))
    print(f"\n[cloud replacement] {counter['i']} updates, {fps:.1f} FPS, p95={p95:.1f}ms, worst={worst:.1f}ms")

    assert fps >= 24.0
    assert p95 <= 67.0
