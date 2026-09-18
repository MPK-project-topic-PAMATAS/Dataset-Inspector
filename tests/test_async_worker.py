"""Tests for the background worker introduced to keep the Qt UI thread
responsive during playback/scrubbing (see SYSTEM_FUNCTIONS.md).

These deliberately do not touch real sensor files: they use small fakes for
SvoReader/RecordingSession so the tests run fast and don't depend on any
particular recording being present on disk. Dataset Inspector remains
read-only throughout -- nothing here writes into a recording directory.
"""
from __future__ import annotations

import numpy as np
import pytest

from worker import FrameRequest, SensorWorker


class FakeSvo:
    """Stands in for SvoReader.read_frame without touching a real .svo2 file."""

    def __init__(self, ok=True, raise_error=False):
        self.ok = ok
        self.raise_error = raise_error
        self.calls = []

    def read_frame(self, index):
        self.calls.append(index)
        if self.raise_error:
            raise RuntimeError("simulated decode failure")
        if not self.ok:
            return {"ok": False}
        return {"ok": True, "left": np.zeros((2, 2, 3), dtype=np.uint8), "right": np.ones((2, 2, 3), dtype=np.uint8)}


class FakeRadarRow:
    """Stands in for a pandas Series row from RecordingSession.radar.

    Only implements what SensorWorker._radar_row_key needs: `row["frame_id"]`
    (preferred) or `row.name` (its fallback), so tests don't need pandas.
    """

    def __init__(self, frame_id=None, name=0):
        self._frame_id = frame_id
        self.name = name

    def __getitem__(self, key):
        if key == "frame_id" and self._frame_id is not None:
            return self._frame_id
        raise KeyError(key)


class FakeSession:
    """Stands in for RecordingSession.load_radar_heatmap/load_radar_points."""

    def __init__(self, heatmap=None, points=None, raise_error=False):
        self.heatmap = heatmap
        self.points = points if points is not None else []
        self.raise_error = raise_error
        self.heatmap_reads = 0
        self.points_reads = 0

    def load_radar_heatmap(self, row):
        self.heatmap_reads += 1
        if self.raise_error:
            raise RuntimeError("simulated radar read failure")
        return self.heatmap

    def load_radar_points(self, row):
        self.points_reads += 1
        if self.raise_error:
            raise RuntimeError("simulated radar read failure")
        return self.points


# --------------------------------------------------------------------------- #
# SensorWorker: direct, in-thread tests (Qt delivers same-thread connections
# synchronously, so these run without needing a real QThread).
# --------------------------------------------------------------------------- #

def test_worker_delivers_camera_and_radar_results():
    worker = SensorWorker()
    camera_results = []
    radar_results = []
    worker.cameraReady.connect(lambda gen, payload: camera_results.append((gen, payload)))
    worker.radarReady.connect(lambda gen, payload: radar_results.append((gen, payload)))

    svo = FakeSvo(ok=True)
    session = FakeSession(heatmap=np.ones((4, 4)), points=[{"x": 1.0, "y": 2.0, "z": 0.0, "velocity_mps": 0.0}])
    request = FrameRequest(
        generation=7, svo=svo, svo_index=42, session=session,
        radar_row=FakeRadarRow(frame_id=50), radar_delta_ns=123,
    )

    worker.process(request)

    assert svo.calls == [42]  # worker used the index the UI thread already resolved
    assert len(camera_results) == 1
    gen, payload = camera_results[0]
    assert gen == 7
    assert payload["left"].shape == (2, 2, 3)
    assert payload["right"].shape == (2, 2, 3)

    assert len(radar_results) == 1
    gen, payload = radar_results[0]
    assert gen == 7
    assert payload["delta_ns"] == 123
    assert payload["points"][0]["x"] == 1.0


def test_worker_camera_exception_does_not_raise_and_reports_error():
    worker = SensorWorker()
    errors = []
    finished = []
    worker.errorOccurred.connect(lambda gen, kind, msg: errors.append((gen, kind, msg)))
    worker.finished.connect(finished.append)

    svo = FakeSvo(raise_error=True)
    request = FrameRequest(generation=3, svo=svo, svo_index=0, session=None, radar_row=None, radar_delta_ns=None)

    # Must not raise -- a broken sensor read must not crash the GUI.
    worker.process(request)

    assert errors == [(3, "camera", "simulated decode failure")]
    assert finished == [3]


def test_worker_radar_exception_does_not_raise_and_reports_error():
    worker = SensorWorker()
    errors = []
    worker.errorOccurred.connect(lambda gen, kind, msg: errors.append((gen, kind, msg)))

    session = FakeSession(raise_error=True)
    request = FrameRequest(
        generation=9, svo=None, svo_index=None, session=session,
        radar_row=FakeRadarRow(frame_id=51), radar_delta_ns=0,
    )

    worker.process(request)

    assert errors == [(9, "radar", "simulated radar read failure")]


def test_worker_camera_decode_not_ok_reports_error_without_raising():
    worker = SensorWorker()
    errors = []
    worker.errorOccurred.connect(lambda gen, kind, msg: errors.append((gen, kind, msg)))

    svo = FakeSvo(ok=False)
    request = FrameRequest(generation=1, svo=svo, svo_index=5, session=None, radar_row=None, radar_delta_ns=None)

    worker.process(request)

    assert errors == [(1, "camera", "frame decode failed")]


def test_worker_finished_always_emitted_once_per_request():
    worker = SensorWorker()
    finished = []
    worker.finished.connect(finished.append)

    request = FrameRequest(generation=11, svo=None, svo_index=None, session=None, radar_row=None, radar_delta_ns=None)
    worker.process(request)

    assert finished == [11]


# --------------------------------------------------------------------------- #
# MainWindow-level staleness / coalescing behavior.
# --------------------------------------------------------------------------- #

@pytest.fixture
def window(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)
    yield win
    win.close()


def test_stale_camera_result_is_ignored(window):
    window._generation = 5
    payload = {"left": np.zeros((2, 2, 3), dtype=np.uint8), "right": np.zeros((2, 2, 3), dtype=np.uint8)}

    assert window.left_label.pixmap() is None or window.left_label.pixmap().isNull()
    window._on_camera_ready(4, payload)  # stale: current generation has moved on to 5

    # A stale (superseded) result must not touch the widget at all -- the
    # label must remain in its untouched (null pixmap) state.
    pix = window.left_label.pixmap()
    assert pix is None or pix.isNull()


def test_fresh_camera_result_is_applied(window):
    window._generation = 5
    payload = {"left": np.zeros((3, 3, 3), dtype=np.uint8), "right": np.ones((3, 3, 3), dtype=np.uint8)}

    window._on_camera_ready(5, payload)  # matches current generation

    assert window.left_label.pixmap() is not None
    assert not window.left_label.pixmap().isNull()


def test_stale_radar_error_does_not_change_title(window):
    window._generation = 5
    window.points_plot.setTitle("frame 4 detections")

    window._on_worker_error(4, "radar", "boom")  # stale

    assert window.points_plot.plotItem.titleLabel.text == "frame 4 detections"


def test_fresh_radar_error_updates_title(window):
    window._generation = 5

    window._on_worker_error(5, "radar", "boom")  # current

    assert "boom" in window.points_plot.plotItem.titleLabel.text


def test_busy_worker_coalesces_to_latest_request_only(window):
    window._worker_busy = True

    req_a = FrameRequest(generation=1, svo=None, svo_index=None, session=None, radar_row=None, radar_delta_ns=None)
    req_b = FrameRequest(generation=2, svo=None, svo_index=None, session=None, radar_row=None, radar_delta_ns=None)

    window._pending_request = req_a
    window._pending_request = req_b  # a second seek before the first finishes replaces, never queues both

    assert window._pending_request is req_b
    assert window._pending_request.generation == 2


def test_worker_finished_dispatches_only_the_coalesced_pending_request(window, monkeypatch):
    sent = []
    monkeypatch.setattr(window, "_send_to_worker", lambda request: sent.append(request))

    pending = FrameRequest(generation=42, svo=None, svo_index=None, session=None, radar_row=None, radar_delta_ns=None)
    window._worker_busy = True
    window._pending_request = pending

    window._on_worker_finished(1)

    assert window._worker_busy is False
    assert window._pending_request is None
    assert sent == [pending]


def test_worker_finished_with_no_pending_request_leaves_worker_idle(window, monkeypatch):
    sent = []
    monkeypatch.setattr(window, "_send_to_worker", lambda request: sent.append(request))

    window._worker_busy = True
    window._pending_request = None

    window._on_worker_finished(1)

    assert window._worker_busy is False
    assert sent == []


# --------------------------------------------------------------------------- #
# Existing nearest-timestamp synchronization must be unchanged: the async
# dispatch must reuse it exactly, not recompute a new policy.
# --------------------------------------------------------------------------- #

def test_dispatch_async_uses_precomputed_sync_index_not_a_fresh_search(window, monkeypatch):
    """Since the sync-index change, _dispatch_async must read the precomputed
    mapping (O(1) array lookup) rather than calling the reference
    nearest_* functions itself -- see sync_index.py."""
    import pandas as pd

    class FakeSvoForSync:
        def nearest_frame_index(self, ts):
            raise AssertionError("dispatch must use the precomputed SyncIndex, not search again")

        def close(self):
            pass

    class FakeSyncIndex:
        def svo_for(self, frame_index):
            assert frame_index == window.current_index
            return 7  # sentinel: whatever the precomputed index says is used verbatim

        def radar_for(self, frame_index):
            assert frame_index == window.current_index
            return 0

    class FakeSessionForSync:
        radar = pd.DataFrame({"timestamp_ns": [555 - 123]})

    window.svo = FakeSvoForSync()
    window.session = FakeSessionForSync()
    window.sync_index = FakeSyncIndex()
    window.current_index = 3
    # This test is specifically about the NEAREST/SyncIndex-based lookup
    # (radar_causal.py's separate CAUSAL policy, the new default, has its
    # own dedicated tests) -- opt into NEAREST explicitly so FakeSyncIndex's
    # radar_for() is actually exercised.
    window._radar_temporal_mode = "NEAREST"

    sent = []
    monkeypatch.setattr(window, "_send_to_worker", lambda request: sent.append(request))

    window._dispatch_async(ts=555)

    assert len(sent) == 1
    request = sent[0]
    # svo_index/radar_row come straight from the precomputed index, and
    # delta_ns is plain arithmetic on the already-located row -- exactly
    # nearest_radar_row's own formula (row.timestamp_ns - query timestamp).
    assert request.svo_index == 7
    assert int(request.radar_row["timestamp_ns"]) == 555 - 123
    assert request.radar_delta_ns == -123
