"""Deterministic, headless performance/resource profiling tests for the
Ouster 3D LiDAR viewer (lidar_3d_view.py).

These measure the *data-path* cost of Lidar3DView.set_points() (buffer
update: numpy finite-guard + color computation + GLScatterPlotItem.setData)
and resource/memory behavior. They do not require a live OpenGL context --
QT_QPA_PLATFORM=offscreen cannot paint a QOpenGLWidget, but it can construct
one and exercise its data path, which is exactly what set_points() is.
Actual GPU frame-cadence (FPS) measurements live in
test_lidar_3d_live_gl.py, clearly separated and skipped unless a real
display is available (see that file for why).

No production code was changed as a result of these measurements: every
measured value here is comfortably inside its acceptance budget (see the
session's performance report). These tests exist to keep it that way.
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np
import pytest

# psutil is already present in this project's environment (a transitive
# dependency); used here only for RSS measurement in these test-only checks.
psutil = pytest.importorskip("psutil")


# --------------------------------------------------------------------------- #
# Shared helpers.
# --------------------------------------------------------------------------- #

def _percentiles(samples_ms):
    arr = np.asarray(samples_ms, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 95)), float(arr.max())


def _synthetic_cloud(n_points: int, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic cloud -- never written under output_multi-record/."""
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    return ((rng.random((n_points, 3), dtype=np.float32) - 0.5) * 40.0)


def _bench_set_points(view, cloud, reps: int, color_mode: str = "Height"):
    view.color_mode_combo.setCurrentText(color_mode)
    times = []
    for i in range(reps):
        t0 = time.perf_counter()
        view.set_points(cloud, scan_index=i, timestamp_ns=i * 1_000_000)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _rss_mb() -> float:
    gc.collect()
    return psutil.Process().memory_info().rss / (1024 * 1024)


@pytest.fixture
def view(qtbot):
    from lidar_3d_view import Lidar3DView

    w = Lidar3DView()
    qtbot.addWidget(w)
    return w


# --------------------------------------------------------------------------- #
# 1A / 7: set_points() latency across representative and stress cloud sizes.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_points", [0, 1, 100, 1_000, 10_000, 50_000, 100_000])
def test_set_points_latency_by_size_is_measured(view, n_points):
    """Not budget-gated (no explicit budget exists for these sizes) --
    purely a measurement, reported via -s, that must complete without error."""
    cloud = _synthetic_cloud(n_points, seed=n_points)
    times = _bench_set_points(view, cloud, reps=20)
    median, p95, worst = _percentiles(times)
    print(f"\n[set_points n={n_points}] median={median:.3f}ms p95={p95:.3f}ms max={worst:.3f}ms")


def test_set_points_latency_typical_cloud_meets_budget(view):
    """Typical recorded Ouster scan in this dataset is ~3,800-4,000 points
    after the existing LidarReader downsampling -- see SYSTEM_FUNCTIONS.md.
    Budget (spec Section 2): median <=20ms target / p95 <=75ms acceptance /
    p95 >100ms hard failure."""
    cloud = _synthetic_cloud(4_000, seed=4000)
    times = _bench_set_points(view, cloud, reps=30)
    median, p95, worst = _percentiles(times)
    print(f"\n[typical 4000-pt cloud] median={median:.3f}ms p95={p95:.3f}ms max={worst:.3f}ms")
    assert p95 <= 100.0, f"HARD FAILURE budget violated: p95={p95:.2f}ms > 100ms"
    assert p95 <= 75.0, f"acceptance budget violated: p95={p95:.2f}ms > 75ms"


def test_set_points_latency_150k_cloud_meets_budget(view):
    """Budget (spec Section 2): median <=50ms target / p95 <=150ms
    acceptance / p95 >250ms hard failure."""
    cloud = _synthetic_cloud(150_000, seed=150_000)
    times = _bench_set_points(view, cloud, reps=15)
    median, p95, worst = _percentiles(times)
    print(f"\n[150k-pt cloud] median={median:.3f}ms p95={p95:.3f}ms max={worst:.3f}ms")
    assert p95 <= 250.0, f"HARD FAILURE budget violated: p95={p95:.2f}ms > 250ms"
    assert p95 <= 150.0, f"acceptance budget violated: p95={p95:.2f}ms > 150ms"


# --------------------------------------------------------------------------- #
# 1B: repeated update latency (>=100 consecutive calls).
# --------------------------------------------------------------------------- #

def test_repeated_updates_100x_on_typical_cloud(view):
    cloud = _synthetic_cloud(4_000, seed=1)
    times = _bench_set_points(view, cloud, reps=100)
    median, p95, worst = _percentiles(times)
    print(f"\n[100x repeated, typical cloud] median={median:.3f}ms p95={p95:.3f}ms max={worst:.3f}ms")
    assert p95 <= 75.0


# --------------------------------------------------------------------------- #
# 1D / 5: memory behavior. Thresholds are deliberately generous relative to
# the spec's 5%/10% figures -- RSS on a shared/virtualized CI box is noisy
# (allocator/page-cache behavior), so the bar here is "no large, sustained
# growth", not a tight percentage; the *shape* of growth (plateauing, not
# climbing) is checked explicitly via test_memory_plateaus_not_climbs below,
# which is the more reliable signal for "no leak".
# --------------------------------------------------------------------------- #

def test_memory_bounded_after_100_and_1000_updates(view):
    baseline = _rss_mb()
    cloud_gen = lambda i: _synthetic_cloud(4_000, seed=i)  # noqa: E731

    for i in range(100):
        view.set_points(cloud_gen(i), scan_index=i, timestamp_ns=i)
    after_100 = _rss_mb()

    for i in range(100, 1_000):
        view.set_points(cloud_gen(i), scan_index=i, timestamp_ns=i)
    after_1000 = _rss_mb()

    growth_100_pct = (after_100 - baseline) / baseline * 100
    growth_1000_pct = (after_1000 - baseline) / baseline * 100
    print(f"\n[memory] baseline={baseline:.1f}MB after100={after_100:.1f}MB "
          f"(+{growth_100_pct:.1f}%) after1000={after_1000:.1f}MB (+{growth_1000_pct:.1f}%)")

    # A genuine per-frame leak at 4000 points/update would show up as tens
    # of MB within 1000 iterations; this catches that class of regression
    # without being sensitive to a few MB of one-time allocator warm-up.
    assert growth_1000_pct < 25.0, f"unbounded memory growth suspected: +{growth_1000_pct:.1f}% after 1000 updates"


def test_memory_plateaus_not_climbs_across_alternating_sizes(view):
    """The real signal for 'no leak, no historical retention': repeating the
    same alternating workload many times must not keep climbing -- it must
    plateau after an initial (one-time) allocator warm-up."""
    small = _synthetic_cloud(100, seed=10)
    big = _synthetic_cloud(100_000, seed=11)

    def run_block():
        for i in range(100):
            view.set_points(small if i % 2 == 0 else big, scan_index=i, timestamp_ns=i)
        return _rss_mb()

    readings = [run_block() for _ in range(5)]
    print(f"\n[alternating small/large, 5 blocks of 100] RSS readings: {[f'{r:.1f}' for r in readings]}")

    # Compare the growth across the LAST two blocks (after warm-up) to the
    # growth across the FIRST two -- a real leak keeps growing at a similar
    # (or increasing) rate every block; a one-time warm-up does not.
    early_growth = readings[1] - readings[0]
    late_growth = readings[-1] - readings[-2]
    assert late_growth <= max(early_growth, 1.0), (
        f"memory still growing after warm-up (early block growth={early_growth:.2f}MB, "
        f"late block growth={late_growth:.2f}MB) -- possible leak"
    )


def test_missing_lidar_and_clear_do_not_leak_across_many_cycles(view):
    baseline = None
    for cycle in range(200):
        view.set_points(_synthetic_cloud(2000, seed=cycle), scan_index=cycle, timestamp_ns=cycle)
        view.show_missing("No LiDAR data for this frame")
        view.clear()
        if cycle == 10:
            baseline = _rss_mb()
    after = _rss_mb()
    growth_pct = (after - baseline) / baseline * 100
    print(f"\n[200x set/missing/clear cycles] growth after warm-up: {growth_pct:.1f}%")
    assert growth_pct < 15.0


# --------------------------------------------------------------------------- #
# 1F / 6: GL/Qt rendering-object stability -- exact GPU memory is not
# reliably measurable without a vendor-specific API (not used here, per
# spec Section 6); object-count stability in the GLViewWidget's own item
# list is the observable, portable substitute.
# --------------------------------------------------------------------------- #

def test_gl_view_item_count_stable_across_1000_updates(view):
    initial_items = len(view.gl_view.items)
    initial_scatter_id = id(view.scatter)
    initial_axis_id = id(view.axis_item)

    for i in range(1_000):
        view.set_points(_synthetic_cloud(1_000, seed=i % 7), scan_index=i, timestamp_ns=i)

    print(f"\n[GL items] before={initial_items} after={len(view.gl_view.items)}")
    assert len(view.gl_view.items) == initial_items
    assert id(view.scatter) == initial_scatter_id
    assert id(view.axis_item) == initial_axis_id


def test_gl_view_item_count_stable_across_missing_and_clear_cycles(view):
    initial_items = len(view.gl_view.items)
    for i in range(200):
        view.set_points(_synthetic_cloud(500, seed=i), scan_index=i, timestamp_ns=i)
        view.show_missing("No LiDAR data for this frame")
        view.clear()
    assert len(view.gl_view.items) == initial_items


# --------------------------------------------------------------------------- #
# 8: NaN/Inf path -- repeated invalid updates must not leak or mutate input.
# --------------------------------------------------------------------------- #

def test_repeated_nan_inf_updates_do_not_leak_or_mutate_source(view):
    baseline = None
    for cycle in range(300):
        pts = _synthetic_cloud(2000, seed=cycle)
        pts[::7] = np.nan
        pts[::11] = np.inf
        original = pts.copy()
        view.set_points(pts, scan_index=cycle, timestamp_ns=cycle)
        assert np.array_equal(pts, original, equal_nan=True)  # never mutated
        if cycle == 20:
            baseline = _rss_mb()
    after = _rss_mb()
    growth_pct = (after - baseline) / baseline * 100
    print(f"\n[300x NaN/Inf updates] growth after warm-up: {growth_pct:.1f}%")
    assert growth_pct < 15.0


def test_nan_inf_displayed_point_count_is_correct():
    from PySide6.QtWidgets import QApplication
    from lidar_3d_view import Lidar3DView

    app = QApplication.instance() or QApplication(sys.argv)
    v = Lidar3DView()
    pts = np.array([[0, 0, 0], [np.nan, 0, 0], [0, np.inf, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    v.set_points(pts, scan_index=0, timestamp_ns=0)
    assert len(v._current_points) == 3
    assert "3 points" in v.header_label.text()
    assert "2 non-finite excluded" in v.header_label.text()


# --------------------------------------------------------------------------- #
# 9 / 10: color-mode and point-size change performance -- rendering-only.
# --------------------------------------------------------------------------- #

def test_color_mode_switch_performance_and_no_derived_array_accumulation(view):
    cloud = _synthetic_cloud(100_000, seed=99)
    view.set_points(cloud, scan_index=0, timestamp_ns=0)

    times = []
    for i in range(40):
        mode = "Height" if i % 2 == 0 else "Uniform"
        t0 = time.perf_counter()
        view.color_mode_combo.setCurrentText(mode)
        times.append((time.perf_counter() - t0) * 1000)
    median, p95, worst = _percentiles(times)
    print(f"\n[color mode switch x40, 100k cloud] median={median:.3f}ms p95={p95:.3f}ms max={worst:.3f}ms")
    assert p95 <= 100.0  # generous sanity bound; no explicit spec budget for this operation

    # The underlying point buffer itself is never touched by a color-mode change.
    assert len(view._current_points) == 100_000


def test_point_size_change_performance_on_large_cloud(view):
    cloud = _synthetic_cloud(150_000, seed=7)
    view.set_points(cloud, scan_index=0, timestamp_ns=0)

    before_items = len(view.gl_view.items)
    times = []
    for size in list(range(1, 11)) * 4:
        t0 = time.perf_counter()
        view.point_size_slider.setValue(size)
        times.append((time.perf_counter() - t0) * 1000)
    median, p95, worst = _percentiles(times)
    print(f"\n[point size change x40, 150k cloud] median={median:.3f}ms p95={p95:.3f}ms max={worst:.3f}ms")
    assert p95 <= 100.0
    assert len(view.gl_view.items) == before_items  # no rebuild of unrelated objects


# --------------------------------------------------------------------------- #
# 11 / 12: camera-control responsiveness and no I/O during interaction.
# --------------------------------------------------------------------------- #

def test_rapid_camera_control_stress_has_no_io_and_stays_responsive(view, monkeypatch):
    import builtins

    cloud = _synthetic_cloud(100_000, seed=3)
    view.set_points(cloud, scan_index=0, timestamp_ns=0)

    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("camera interaction must never touch disk")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        times = []
        for i in range(200):
            t0 = time.perf_counter()
            if i % 4 == 0:
                view.gl_view.orbit(5, 2)
            elif i % 4 == 1:
                view.gl_view.opts["distance"] *= 0.98
            elif i % 4 == 2:
                view.gl_view.pan(1, 1, 0, relative="view-upright")
            else:
                view.reset_view() if i % 8 == 3 else view.fit_to_cloud()
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        monkeypatch.setattr(builtins, "open", real_open)

    median, p95, worst = _percentiles(times)
    print(f"\n[camera control stress x200] median={median:.3f}ms p95={p95:.3f}ms max={worst:.3f}ms")
    assert p95 <= 50.0
