"""Integration tests for MainWindow's radar 2D/3D toggle and NEAREST/CAUSAL
temporal display policy (see radar_causal.py for the pure-function tests of
the CAUSAL rule itself).

Uses the same minimal-fakes-through-a-real-MainWindow approach as
test_lifecycle_stress.py (FakeSession/FakeSvo/FakeSyncIndex + a real
SessionLoadResult committed via _on_session_load_finished), so every
existing invariant (generation protection, worker/cache reuse, etc.)
applies exactly as it does for real sessions.

Worked example used throughout (matches the design brief's own numbers):

    camera timestamps (ms): [80, 100, 200, 300]
    radar timestamps   (ms): [90, 150, 220, 310]

    frame 0 (t=80):  CAUSAL -> no sample yet   NEAREST -> 90  (future, +10ms)
    frame 1 (t=100): CAUSAL -> 90 (-10ms)       NEAREST -> 90  (-10ms, same)
    frame 2 (t=200): CAUSAL -> 150 (-50ms)      NEAREST -> 220 (future, +20ms)
    frame 3 (t=300): CAUSAL -> 220 (-80ms)      NEAREST -> 310 (future, +10ms)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensor_health import SensorHealthModel
from session_loader import SessionLoadResult
from sync_index import MISSING

CAMERA_MS = [80, 100, 200, 300]
RADAR_MS = [90, 150, 220, 310]


class FakeSvo:
    def __init__(self, tag="svo"):
        self.closed = False
        self.tag = tag
        self.timestamps_ns = np.array([0], dtype=np.int64)
        self.has_zed_imu = False

    def close(self):
        self.closed = True

    def nearest_frame_index(self, ts):
        return 0

    def read_frame(self, index):
        return {"ok": True, "left": np.zeros((2, 2, 3), dtype=np.uint8), "right": np.zeros((2, 2, 3), dtype=np.uint8)}


class FakeSyncIndex:
    """NEAREST radar_for() hardcoded to match the worked example above --
    everything else (lidar/mti/svo) reports MISSING, matching an
    otherwise-empty session."""

    _NEAREST = {0: 0, 1: 0, 2: 2, 3: 3}

    def radar_for(self, i):
        return self._NEAREST[i]

    def lidar_for(self, i):
        return MISSING

    def mti_for(self, i):
        return MISSING

    def svo_for(self, i):
        return MISSING


class FakeSession:
    def __init__(self, session_id="S"):
        self.session_id = session_id
        self.path = Path("/nonexistent") / session_id
        self.frames = pd.DataFrame({"timestamp_ns": np.array([t * 1_000_000 for t in CAMERA_MS], dtype=np.int64)})
        self.radar = pd.DataFrame({
            "frame_id": [0, 1, 2, 3],
            "timestamp_ns": np.array([t * 1_000_000 for t in RADAR_MS], dtype=np.int64),
            "heatmap_path": ["", "", "", ""],
            "points_path": ["", "", "", ""],
        })
        self.events = pd.DataFrame()
        self.mti = None
        self.lidar = None
        self.lidar_error = None
        self.manifest = {}
        self.session_manifest = {}

    @property
    def svo_path(self):
        return self.path / "raw" / "recording.svo2"

    def load_radar_heatmap(self, row):
        return None

    def load_radar_points(self, row):
        return []


def make_result(generation, session_id="S"):
    session = FakeSession(session_id)
    sync_index = FakeSyncIndex()
    return SessionLoadResult(
        generation=generation, session_dir=session.path, ok=True,
        session=session, svo=FakeSvo(tag=session_id),
        sync_index=sync_index, validation_result=None,
        sensor_health=SensorHealthModel(session, sync_index),
    )


@pytest.fixture
def window(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)
    yield win
    win._worker_busy = False
    win._pending_request = None
    win.close()


def settle(window):
    window._worker_busy = False
    window._pending_request = None


def load(window, generation=1, session_id="S"):
    window._session_generation = generation
    window._on_session_load_finished(make_result(generation, session_id))
    settle(window)


# --------------------------------------------------------------------------- #
# Part 1: 2D/3D toggle.
# --------------------------------------------------------------------------- #

def test_both_widgets_constructed_once_and_2d_is_the_initial_default(window):
    assert window.points_plot is not None
    assert window.radar_3d_view is not None
    assert window.radar_view_stack.count() == 2
    assert window._radar_view_mode == "2D"
    assert window.radar_view_stack.currentWidget() is window.points_plot


def test_toggle_2d_to_3d_and_back_does_not_recreate_widgets(window, monkeypatch):
    load(window)
    points_plot_id = id(window.points_plot)
    radar_3d_id = id(window.radar_3d_view)
    gl_view_id = id(window.radar_3d_view.gl_view)

    dispatch_calls = []
    monkeypatch.setattr(window, "_dispatch_async", lambda *a, **k: dispatch_calls.append(a))

    window._on_radar_view_mode_changed("3D")
    assert window.radar_view_stack.currentWidget() is window.radar_3d_view
    window._on_radar_view_mode_changed("2D")
    assert window.radar_view_stack.currentWidget() is window.points_plot

    assert id(window.points_plot) == points_plot_id
    assert id(window.radar_3d_view) == radar_3d_id
    assert id(window.radar_3d_view.gl_view) == gl_view_id
    # Presentation-only: switching visualization mode must never re-dispatch.
    assert dispatch_calls == []


def test_repeated_toggling_preserves_sample_timestamp_delta_and_health_state(window):
    load(window)
    window._seek(2)  # camera=200ms -> CAUSAL sample at 150ms (default mode)
    header_before = window.radar_header_label.text()
    health_before = window._current_health_snapshot.radar.state

    for _ in range(20):
        window._on_radar_view_mode_changed("3D")
        window._on_radar_view_mode_changed("2D")

    assert window.radar_header_label.text() == header_before
    assert window._current_health_snapshot.radar.state == health_before
    assert window.current_index == 2  # no playback reset


def test_toggle_updates_the_view_mode_line_in_the_header(window):
    # Regression: an earlier implementation replayed a pre-rendered header
    # string verbatim on toggle, so the "2D"/"3D" line never actually
    # updated even though the widget underneath it correctly switched.
    load(window)
    window._seek(1)
    assert window.radar_header_label.text().splitlines()[1] == "2D"
    window._on_radar_view_mode_changed("3D")
    assert window.radar_header_label.text().splitlines()[1] == "3D"
    window._on_radar_view_mode_changed("2D")
    assert window.radar_header_label.text().splitlines()[1] == "2D"


def test_toggle_never_reloads_session(window, monkeypatch):
    load(window)
    session_before = window.session
    load_calls = []
    monkeypatch.setattr(window, "_load_session", lambda *a, **k: load_calls.append(a))

    window._on_radar_view_mode_changed("3D")
    window._on_radar_view_mode_changed("2D")

    assert window.session is session_before
    assert load_calls == []


# --------------------------------------------------------------------------- #
# Part 2: NEAREST vs CAUSAL selection (design brief's worked example).
# --------------------------------------------------------------------------- #

def test_default_temporal_mode_is_causal(window):
    assert window._radar_temporal_mode == "CAUSAL"


def test_causal_frame1_selects_90ms_with_negative_delta(window):
    load(window)
    window._seek(1)  # camera=100ms
    row, delta_ns, is_future, no_causal_yet = window._resolve_radar_selection(100_000_000)
    assert no_causal_yet is False
    assert int(row["timestamp_ns"]) == 90_000_000
    assert delta_ns == -10_000_000
    assert is_future is False


def test_causal_frame2_selects_150ms_not_the_nearest_220ms(window):
    load(window)
    window._seek(2)  # camera=200ms
    row, delta_ns, is_future, no_causal_yet = window._resolve_radar_selection(200_000_000)
    assert int(row["timestamp_ns"]) == 150_000_000  # NOT 220ms (that's NEAREST's pick)
    assert delta_ns == -50_000_000
    assert is_future is False
    assert no_causal_yet is False


def test_causal_frame0_before_first_radar_sample_yields_no_sample_yet(window):
    """Mandatory case: NEAREST would pick 90ms (a future sample from frame
    0's perspective), but CAUSAL must show "no sample yet" instead of
    falling back to it."""
    load(window)
    window._seek(0)  # camera=80ms
    row, delta_ns, is_future, no_causal_yet = window._resolve_radar_selection(80_000_000)
    assert row is None
    assert delta_ns is None
    assert no_causal_yet is True
    assert "No causal sample yet" in window.radar_header_label.text()
    # Honest empty state, not stale/fabricated data:
    assert window.points_scatter.data is not None and len(window.points_scatter.data) == 0
    assert len(window.radar_3d_view._current_points) == 0


def test_nearest_frame0_selects_90ms_and_flags_it_as_a_future_sample(window):
    load(window)
    window._radar_temporal_mode = "NEAREST"
    window._seek(0)  # camera=80ms
    row, delta_ns, is_future, no_causal_yet = window._resolve_radar_selection(80_000_000)
    assert int(row["timestamp_ns"]) == 90_000_000
    assert delta_ns == 10_000_000
    assert is_future is True
    assert no_causal_yet is False
    assert "Future sample" in window.radar_header_label.text()


def test_switching_temporal_mode_changes_displayed_sample_without_reload_or_seek(window, monkeypatch):
    load(window)
    window._seek(2)  # camera=200ms
    session_before = window.session
    index_before = window.current_index

    seek_calls = []
    monkeypatch.setattr(window, "_seek", lambda *a, **k: seek_calls.append(a))

    window._on_radar_temporal_mode_changed("NEAREST")
    row, delta_ns, is_future, _ = window._resolve_radar_selection(200_000_000)
    assert int(row["timestamp_ns"]) == 220_000_000
    assert is_future is True

    window._on_radar_temporal_mode_changed("CAUSAL")
    row, delta_ns, is_future, _ = window._resolve_radar_selection(200_000_000)
    assert int(row["timestamp_ns"]) == 150_000_000
    assert is_future is False

    assert window.session is session_before
    assert window.current_index == index_before
    assert seek_calls == []  # temporal mode toggle must never call _seek


def test_sensor_health_state_is_never_mutated_by_temporal_mode(window):
    """SensorHealthModel is always NEAREST-based and must report the exact
    same state regardless of the display-only CAUSAL/NEAREST toggle."""
    load(window)
    window._seek(2)
    state_causal = window._current_health_snapshot.radar.state
    window._on_radar_temporal_mode_changed("NEAREST")
    state_nearest = window._current_health_snapshot.radar.state
    assert state_causal == state_nearest  # unaffected by the display toggle


# --------------------------------------------------------------------------- #
# Playback-order regression: displayed radar timestamp must never exceed
# the camera timestamp in CAUSAL mode, across every frame visited by normal
# advance (speed only changes the timer interval between _seek calls, never
# which frames get visited -- see MainWindow._advance/_start_playback).
# --------------------------------------------------------------------------- #

def test_causal_invariant_holds_across_every_frame_in_sequence(window):
    load(window)
    assert window._radar_temporal_mode == "CAUSAL"
    for i in range(len(CAMERA_MS)):
        window._seek(i)
        camera_ts = CAMERA_MS[i] * 1_000_000
        row, delta_ns, is_future, no_causal_yet = window._resolve_radar_selection(camera_ts)
        if row is None:
            assert no_causal_yet is True
        else:
            assert int(row["timestamp_ns"]) <= camera_ts
            assert is_future is False


def test_causal_invariant_holds_reverse_scrubbing(window):
    load(window)
    for i in reversed(range(len(CAMERA_MS))):
        window._seek(i)
        camera_ts = CAMERA_MS[i] * 1_000_000
        row, _, is_future, no_causal_yet = window._resolve_radar_selection(camera_ts)
        if row is not None:
            assert int(row["timestamp_ns"]) <= camera_ts
            assert is_future is False


@pytest.mark.parametrize("speed", [1.0, 2.0, 4.0])
def test_causal_invariant_holds_during_real_playback_at_each_speed(window, qtbot, speed):
    load(window)
    window.speed = speed
    window._seek(0)
    window._start_playback()
    qtbot.waitUntil(lambda: window.current_index >= len(CAMERA_MS) - 1 or not window.playing, timeout=3000)
    window._stop_playback()
    # Whatever frame we ended up on, the causal invariant must still hold.
    camera_ts = CAMERA_MS[window.current_index] * 1_000_000
    row, _, is_future, no_causal_yet = window._resolve_radar_selection(camera_ts)
    if row is not None:
        assert int(row["timestamp_ns"]) <= camera_ts
        assert is_future is False


# --------------------------------------------------------------------------- #
# Header text (design brief §15) and no-fabrication checks.
# --------------------------------------------------------------------------- #

def test_header_shows_view_mode_temporal_mode_sample_and_delta(window):
    load(window)
    window._seek(1)
    text = window.radar_header_label.text()
    assert "RADAR" in text
    assert window._radar_view_mode in text
    assert window._radar_temporal_mode in text
    assert "Sample" in text
    assert "Δt" in text


def test_no_causal_yet_never_shows_a_future_timestamp_value(window):
    load(window)
    window._seek(0)
    text = window.radar_header_label.text()
    assert "No causal sample yet" in text
    assert "90" not in text.replace("RADAR", "")  # the 90ms future sample must never leak into the header
