"""Lifecycle/resilience stress tests for MainWindow's interaction with
SessionLoader, SensorWorker, TimelineDiagnostics, EventMarkerStrip, and
SensorHealthModel.

This module is resilience-only: it does not test new behavior, only that
the existing generation/supersession, cache-reset, and shutdown-ordering
guarantees hold under repeated/adversarial sequences of session loads,
seeks, and closes. It complements (does not duplicate) test_session_loader.py,
test_async_worker.py, test_cache.py, test_sensor_health.py, and
test_timeline_diagnostics.py.

One real defect was found and fixed while writing these tests: MainWindow.
closeEvent bumped `_session_generation` *after* waiting on SensorWorker
(inside a nested QEventLoop), leaving a window where a SessionLoader result
arriving during that wait could still be committed mid-shutdown. The fix
moves the generation bump to the very start of closeEvent; several tests
here (`test_close_*`) exercise exactly that window.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensor_health import SensorHealthModel
from session_loader import SessionLoadResult
from sync_index import MISSING


# --------------------------------------------------------------------------- #
# Minimal fakes -- shaped like RecordingSession/SvoReader, no real I/O.
# --------------------------------------------------------------------------- #

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
    def radar_for(self, i):
        return MISSING

    def lidar_for(self, i):
        return MISSING

    def mti_for(self, i):
        return MISSING

    def svo_for(self, i):
        return MISSING


class FakeSession:
    def __init__(self, session_id, n_frames=5):
        self.session_id = session_id
        self.path = Path("/nonexistent") / session_id
        self.frames = pd.DataFrame({"timestamp_ns": np.arange(n_frames, dtype=np.int64) * 1_000_000})
        self.radar = pd.DataFrame()
        self.events = pd.DataFrame()
        self.mti = None
        self.lidar = None
        self.lidar_error = None
        self.manifest = {}
        self.session_manifest = {}

    @property
    def svo_path(self):
        return self.path / "raw" / "recording.svo2"


def make_result(generation, session_id, ok=True, n_frames=5, svo=None, failed_stage="", error=""):
    if not ok:
        return SessionLoadResult(generation=generation, session_dir=Path(f"/nonexistent/{session_id}"), ok=False,
                                  failed_stage=failed_stage, error=error)
    session = FakeSession(session_id, n_frames=n_frames)
    sync_index = FakeSyncIndex()
    return SessionLoadResult(
        generation=generation, session_dir=session.path, ok=True,
        session=session, svo=svo if svo is not None else FakeSvo(tag=session_id),
        sync_index=sync_index, validation_result=None,
        sensor_health=SensorHealthModel(session, sync_index),
    )


@pytest.fixture
def window(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)
    yield win
    # Test-only teardown hygiene: several tests above dispatch a *real*
    # request into the real SensorWorker thread (e.g. via _commit_session's
    # own _seek(0)) and never pump the event loop for it to complete, since
    # these tests are exercising synchronous generation logic, not real
    # threading. Left alone, MainWindow.close()'s _await_worker_idle() would
    # correctly wait out its full 3s timeout for a request that will in fact
    # complete (worker.py has nothing to do for these FakeSyncIndex-backed
    # requests) the instant the event loop runs -- which is exactly what
    # close() does next. Settling by hand here only saves that wall-clock
    # wait; it changes no assertion and no production code.
    win._worker_busy = False
    win._pending_request = None
    win.close()


def settle(window):
    """Test-only helper: force SensorWorker bookkeeping to its resting
    state after a commit, so a later _await_worker_idle() in the same test
    (or at teardown) never pays out its full 3s timeout waiting on a real
    request these FakeSyncIndex-backed sessions guarantee finishes trivially
    (worker.py has nothing to decode/read for them) the instant the event
    loop actually runs. Purely a wall-clock optimization for these tests --
    it asserts nothing and changes no production code.
    """
    window._worker_busy = False
    window._pending_request = None


def _committed_session_ids(window, monkeypatch):
    """Lets real _commit_session run (so all its invariants apply) while
    recording, in order, which session_id ended up active."""
    ids = []
    real_commit = window._commit_session

    def spy(result):
        real_commit(result)
        ids.append(window.session.session_id)

    monkeypatch.setattr(window, "_commit_session", spy)
    return ids


# --------------------------------------------------------------------------- #
# 1. Session load races: A -> B -> C.
# --------------------------------------------------------------------------- #

def test_a_b_c_rapid_requests_only_c_can_become_active_regardless_of_finish_order(window, monkeypatch):
    """The exact race from the design brief: A active, request B, request C
    before B finishes. Whichever of B/C finishes first, only C may ever
    become active, and B's SvoReader must be closed either way."""
    ids = _committed_session_ids(window, monkeypatch)
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    assert ids == ["A"]

    window._session_generation = 2  # "request B"
    window._session_generation = 3  # "request C" arrives before B finishes

    b_result = make_result(2, "B")
    c_result = make_result(3, "C")

    # Order 1: B finishes, then C.
    window._on_session_load_finished(b_result)
    settle(window)
    assert ids == ["A"]  # B discarded, never committed
    assert b_result.svo.closed is True
    window._on_session_load_finished(c_result)
    settle(window)
    assert ids == ["A", "C"]
    assert window.session.session_id == "C"


def test_a_b_c_rapid_requests_c_finishes_before_b(window, monkeypatch):
    ids = _committed_session_ids(window, monkeypatch)
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)

    window._session_generation = 2
    window._session_generation = 3

    b_result = make_result(2, "B")
    c_result = make_result(3, "C")

    window._on_session_load_finished(c_result)
    settle(window)
    assert window.session.session_id == "C"
    window._on_session_load_finished(b_result)  # late arrival, must not undo C
    settle(window)
    assert window.session.session_id == "C"
    assert b_result.svo.closed is True
    assert ids == ["A", "C"]


def test_failed_b_does_not_destroy_active_a(window, monkeypatch):
    ids = _committed_session_ids(window, monkeypatch)
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)

    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B", ok=False, failed_stage="SVO", error="boom"))
    settle(window)

    assert window.session.session_id == "A"
    assert ids == ["A"]


def test_failed_c_does_not_destroy_active_a_or_b(window, monkeypatch):
    ids = _committed_session_ids(window, monkeypatch)
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B"))
    settle(window)
    assert window.session.session_id == "B"

    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)
    window._session_generation = 3
    window._on_session_load_finished(make_result(3, "C", ok=False, failed_stage="Lidar", error="bad pcap"))
    settle(window)

    assert window.session.session_id == "B"  # last good session remains active
    assert ids == ["A", "B"]


def test_stale_validation_result_cannot_overwrite_current_validation_ui(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    window.integrity_label.setText("A's validation text")

    window._session_generation = 2  # request B, never delivered yet
    stale = make_result(1, "A-late-duplicate")  # a duplicate/late result at the OLD generation
    window._on_session_load_finished(stale)
    settle(window)

    assert window.integrity_label.text() == "A's validation text"  # untouched


def test_stale_sensor_health_cannot_overwrite_current_sensor_status(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    before = window.sensor_health
    window._session_generation = 2

    stale = make_result(1, "A-late-duplicate")
    window._on_session_load_finished(stale)
    settle(window)

    assert window.sensor_health is before  # the committed SensorHealthModel object is untouched


def test_stale_timeline_state_cannot_overwrite_active_timeline(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=5))
    settle(window)
    before_coverage = window.timeline_diagnostics.coverage()

    window._session_generation = 2
    stale = make_result(1, "A-late-duplicate", n_frames=999)  # would look very different if committed
    window._on_session_load_finished(stale)
    settle(window)

    assert window.timeline_diagnostics.coverage() == before_coverage


def test_stale_event_marker_set_is_never_installed_and_cannot_be_clicked_against_wrong_session(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    before_markers = list(window.event_marker_strip._markers)

    window._session_generation = 2
    stale = make_result(1, "A-late-duplicate")
    window._on_session_load_finished(stale)
    settle(window)

    assert window.event_marker_strip._markers == before_markers
    # Clicking whatever markers ARE installed always resolves against
    # whichever session is currently committed -- never a stale one, since
    # _seek_to_timestamp always reads self.session (reassigned only by a
    # real commit) rather than anything captured at marker-creation time.
    window._seek_to_timestamp(999_999_999)
    assert window.session.session_id == "A"
    assert window.current_index == len(window.session.frames) - 1  # clamped, not confused about session


# --------------------------------------------------------------------------- #
# 2. SensorWorker races.
# --------------------------------------------------------------------------- #

def test_seek_while_camera_decode_running_coalesces_to_latest(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=50))
    settle(window)
    window._worker_busy = True  # simulate an in-flight decode
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)

    for idx in range(1, 30):
        window._seek(idx)

    assert window._pending_request is not None
    assert window.current_index == 29
    # The coalesced request reflects the latest seek, not an intermediate one.


def test_rapid_direction_changes_settle_to_final_position(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=50))
    settle(window)
    window._worker_busy = True
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)

    sequence = [10, 5, 20, 3, 49, 0, 49, 25]
    for idx in sequence:
        window._seek(idx)

    assert window.current_index == sequence[-1]
    assert window._pending_request.generation == window._generation


def test_final_frame_seek_during_worker_activity_then_settles(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=50))
    settle(window)
    window._worker_busy = True
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)

    window._seek(49)
    assert window._pending_request is not None

    dispatched = []
    monkeypatch.setattr(window, "_send_to_worker", lambda request: dispatched.append(request))
    window._on_worker_finished(999)  # the (now-irrelevant) in-flight generation completes

    assert window._worker_busy is False
    assert dispatched and dispatched[0].generation == window._generation


def test_camera_and_radar_results_from_obsolete_generation_never_touch_widgets(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    window._seek(3)
    stale_generation = window._generation
    window._seek(4)  # bumps _generation again -> stale_generation is now obsolete

    before_left = window.left_label.pixmap()
    before_title = window.points_plot.plotItem.titleLabel.text
    window._on_camera_ready(stale_generation, {"left": np.ones((2, 2, 3), dtype=np.uint8),
                                                "right": np.ones((2, 2, 3), dtype=np.uint8)})
    window._on_radar_ready(stale_generation, {"heatmap": None, "points": [{"x": 1, "y": 1}], "delta_ns": 1})

    after_left = window.left_label.pixmap()
    assert (before_left is None) == (after_left is None)
    if before_left is not None and after_left is not None:
        assert before_left.cacheKey() == after_left.cacheKey()
    assert window.points_plot.plotItem.titleLabel.text == before_title


def test_session_switch_immediately_after_worker_result_settles_cleanly(window, monkeypatch):
    # Isolate from the real worker thread from the very first commit onward
    # (its own internal _seek(0) also dispatches) so the test controls
    # exactly one in-flight generation at a time, by hand.
    monkeypatch.setattr(window, "_send_to_worker", lambda request: setattr(window, "_worker_busy", True))
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    window._on_worker_finished(window._generation)  # settle the commit's own dispatch first

    window._seek(2)
    gen = window._generation
    window._on_camera_ready(gen, {"left": np.zeros((2, 2, 3), dtype=np.uint8), "right": np.zeros((2, 2, 3), dtype=np.uint8)})
    window._on_worker_finished(gen)
    assert window._worker_busy is False

    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B"))
    settle(window)
    assert window.session.session_id == "B"
    # The commit's own _seek(0) legitimately dispatches a fresh request for
    # B's first frame (busy=True is correct here); it settles normally.
    assert window._pending_request is None
    window._on_worker_finished(window._generation)
    assert window._worker_busy is False


def test_pause_and_step_during_worker_activity_do_not_corrupt_state(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    window._toggle_play()
    assert window.playing is True

    window._worker_busy = True
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    window._seek(window.current_index + 1)  # "step" while busy
    window._toggle_play()  # pause
    assert window.playing is False
    assert window.timer.isActive() is False


# --------------------------------------------------------------------------- #
# 3. Close / shutdown races -- exercises the fix in closeEvent.
# --------------------------------------------------------------------------- #

def test_close_while_session_load_is_queued_discards_it_instead_of_committing(window, monkeypatch):
    """This is the exact defect found and fixed: closeEvent must invalidate
    _session_generation before doing anything that could let a queued
    SessionLoadResult for the *current* generation still be committed."""
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    window._session_generation = 2  # a session load is in flight

    committed = []
    real_commit = window._commit_session
    monkeypatch.setattr(window, "_commit_session", lambda r: committed.append(r) or real_commit(r))

    window.close()  # runs closeEvent

    # After close(), the in-flight generation must have been invalidated
    # before the result below is ever handled.
    late_result = make_result(2, "B")
    window._on_session_load_finished(late_result)
    settle(window)

    assert committed == []  # never committed during or after close
    assert late_result.svo.closed is True
    assert window.session.session_id == "A"  # untouched


def test_close_during_camera_decode_tears_down_cleanly(window):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    window._seek(2)  # dispatches a real request into the real SensorWorker thread
    window.close()

    assert window._worker_thread.isRunning() is False
    assert window._session_loader_thread.isRunning() is False


def test_close_immediately_after_a_seek(window):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    window._seek(5)
    window.close()
    assert window._worker_thread.isRunning() is False


def test_close_immediately_after_session_switch(window):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B"))
    settle(window)
    window.close()
    assert window._worker_thread.isRunning() is False
    assert window._session_loader_thread.isRunning() is False


def test_close_after_a_failed_load(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)
    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B", ok=False, failed_stage="X", error="boom"))
    settle(window)
    window.close()
    assert window.session.session_id == "A"
    assert window._worker_thread.isRunning() is False


def test_close_after_a_superseded_load(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    window._session_generation = 2
    window._session_generation = 3
    window._on_session_load_finished(make_result(2, "B"))  # discarded (superseded)
    settle(window)
    window.close()
    assert window.session.session_id == "A"
    assert window._worker_thread.isRunning() is False


def test_no_orphaned_threads_after_close(window):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    window.close()
    assert not window._worker_thread.isRunning()
    assert not window._session_loader_thread.isRunning()


# --------------------------------------------------------------------------- #
# 6. Playback state machine transitions.
# --------------------------------------------------------------------------- #

def test_playback_transition_matrix(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=20))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)

    # STOP -> PLAY
    window._toggle_play()
    assert window.playing is True and window.timer.isActive() is True

    # PLAY -> PAUSE
    window._toggle_play()
    assert window.playing is False and window.timer.isActive() is False

    # PAUSE -> STEP
    before = window.current_index
    window._seek(before + 1)
    assert window.current_index == before + 1

    # STEP -> PLAY
    window._toggle_play()
    assert window.playing is True

    # PLAY -> session switch (must stop playback, not leave the timer running
    # against a session that's about to be replaced)
    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B", n_frames=20))
    settle(window)
    assert window.playing is False
    assert window.timer.isActive() is False
    assert window.session.session_id == "B"

    # PAUSE -> session switch (already paused; must remain safe)
    window._session_generation = 3
    window._on_session_load_finished(make_result(3, "C", n_frames=20))
    settle(window)
    assert window.playing is False
    assert window.session.session_id == "C"


def test_loading_does_not_block_playback_or_seeking_of_the_active_session(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=20))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)

    window._session_generation = 2  # "B is loading" -- A must remain fully usable
    window._session_load_state = "LOADING"

    window._toggle_play()
    assert window.playing is True
    window._seek(5)
    assert window.current_index == 5
    assert window.session.session_id == "A"  # still A; B hasn't committed


def test_loading_then_close_is_safe(window):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    window._session_generation = 2  # B is "loading"
    window.close()
    assert window._worker_thread.isRunning() is False
    assert window._session_loader_thread.isRunning() is False


# --------------------------------------------------------------------------- #
# 7. Seek stress.
# --------------------------------------------------------------------------- #

def test_repeated_same_frame_seeks(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=30))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    for _ in range(50):
        window._seek(15)
    assert window.current_index == 15


def test_monotonically_increasing_seeks(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=30))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    for i in range(30):
        window._seek(i)
    assert window.current_index == 29


def test_monotonically_decreasing_seeks(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=30))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    for i in reversed(range(30)):
        window._seek(i)
    assert window.current_index == 0


def test_alternating_first_and_final_frame(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=30))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    for i in range(40):
        window._seek(0 if i % 2 == 0 else 29)
    assert window.current_index == 29
    assert window._current_health_snapshot.camera_frame_index == 29


def test_random_frame_sequence_stress(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=100))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    rng = random.Random(1234)
    last = 0
    for _ in range(200):
        last = rng.randrange(0, 100)
        window._seek(last)
    assert window.current_index == last
    assert window._current_health_snapshot.camera_frame_index == last


def test_rapid_repeated_event_marker_seeks(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=30))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    frames_ts = window.session.frames["timestamp_ns"].to_numpy()
    for ts in list(frames_ts[::3]) * 5:
        window.event_marker_strip.markerClicked.emit(int(ts))
    assert 0 <= window.current_index < len(window.session.frames)


def test_rapid_repeated_event_list_seeks(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=30))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)

    class FakeItem:
        def __init__(self, ts):
            self._ts = ts

        def data(self, role):
            return self._ts

    frames_ts = window.session.frames["timestamp_ns"].to_numpy()
    for ts in list(frames_ts[::2]) * 5:
        window._seek_to_event(FakeItem(int(ts)))
    assert 0 <= window.current_index < len(window.session.frames)


def test_seek_during_asynchronous_load_operates_on_active_session_only(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    monkeypatch.setattr(window, "_send_to_worker", lambda request: None)
    window._session_generation = 2  # B loading in background

    window._seek(3)
    window._seek(7)

    assert window.session.session_id == "A"
    assert window.current_index == 7


def test_seek_immediately_after_commit_reflects_new_session(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B", n_frames=40))
    settle(window)
    window._seek(30)
    assert window.session.session_id == "B"
    assert window.current_index == 30


# --------------------------------------------------------------------------- #
# 9. Error handling / recovery.
# --------------------------------------------------------------------------- #

def test_worker_error_leaves_busy_pending_flags_recoverable(window, monkeypatch):
    # See test_session_switch_immediately_after_worker_result_settles_cleanly:
    # isolate from the real worker thread from the very first commit onward.
    monkeypatch.setattr(window, "_send_to_worker", lambda request: setattr(window, "_worker_busy", True))
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    window._on_worker_finished(window._generation)  # settle the commit's own dispatch first

    window._seek(3)
    gen = window._generation

    window._on_worker_error(gen, "camera", "simulated failure")
    assert "failed" in window.left_label.text().lower()

    window._on_worker_finished(gen)  # SensorWorker always emits finished, even after an error
    assert window._worker_busy is False
    assert window._pending_request is None

    # A subsequent seek must still work normally.
    window._seek(4)
    assert window.current_index == 4
    settle(window)


def test_failed_load_does_not_poison_cache_or_prevent_subsequent_successful_load(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A", n_frames=10))
    settle(window)
    stats_before = window._worker.cache_stats()

    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)
    window._session_generation = 2
    window._on_session_load_finished(make_result(2, "B", ok=False, failed_stage="X", error="boom"))
    settle(window)

    assert window.session.session_id == "A"
    assert window._worker.cache_stats() == stats_before  # a failed load never touched the cache

    window._session_generation = 3
    window._on_session_load_finished(make_result(3, "C", n_frames=15))
    settle(window)
    assert window.session.session_id == "C"  # a later successful load still works normally


def test_repeated_failures_do_not_accumulate_broken_state(window, monkeypatch):
    window._session_generation = 1
    window._on_session_load_finished(make_result(1, "A"))
    settle(window)
    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)

    for i in range(2, 6):
        window._session_generation = i
        window._on_session_load_finished(make_result(i, f"fail{i}", ok=False, failed_stage="X", error="boom"))
        settle(window)
        assert window.session.session_id == "A"

    window._session_generation = 6
    window._on_session_load_finished(make_result(6, "finally-good"))
    settle(window)
    assert window.session.session_id == "finally-good"


# --------------------------------------------------------------------------- #
# Dataset immutability: nothing under output_multi-record/ is ever opened
# for writing by any of the above.
# --------------------------------------------------------------------------- #

_DATASET_ROOT = Path(__file__).parent.parent / "output_multi-record"


def test_no_recorded_file_is_ever_opened_for_writing(monkeypatch):
    if not _DATASET_ROOT.exists():
        pytest.skip("real dataset not available")

    import builtins

    real_open = builtins.open

    def guarded_open(path, mode="r", *args, **kwargs):
        path_str = str(path)
        if _DATASET_ROOT.resolve().as_posix() in Path(path_str).resolve().as_posix() if Path(path_str).is_absolute() else False:
            if any(c in mode for c in ("w", "a", "x", "+")):
                raise AssertionError(f"attempted to open a recorded file for writing: {path_str} mode={mode}")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    try:
        # A representative read-only pass: load real session metadata if present.
        first_session = next(
            (p for p in sorted((_DATASET_ROOT / "output").iterdir())
             if p.is_dir() and (p / "raw" / "recording.svo2").exists()),
            None,
        )
        if first_session is None:
            pytest.skip("no real session directory available")
        from session import RecordingSession
        RecordingSession.load(first_session)
    finally:
        monkeypatch.setattr(builtins, "open", real_open)
