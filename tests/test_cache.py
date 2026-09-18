"""Tests for the bounded in-memory cache added in front of SensorWorker's
camera decode and radar file reads (see cache.py, worker.py).

These never touch the recorded dataset: SvoReader/RecordingSession are
faked so a cache hit vs. miss can be observed directly by counting calls
into the fakes. The point of this cache is purely to avoid redundant reads/
decodes of data the recording already provided once -- it must never change
what gets displayed, and must never let one session's cached data leak into
another's.
"""
from __future__ import annotations

import numpy as np
import pytest

from cache import CacheConfig, LRUCache
from worker import FrameRequest, SensorWorker


class FakeSvo:
    def __init__(self, tag=0):
        self.calls = []
        self.tag = tag  # lets two FakeSvo instances be told apart even at the same index

    def read_frame(self, index):
        self.calls.append(index)
        # Distinct content per (tag, index), so a wrong cache hit would be visible.
        return {
            "ok": True,
            "left": np.full((2, 2, 3), self.tag * 100 + index, dtype=np.uint8),
            "right": np.full((2, 2, 3), self.tag * 100 + index + 1, dtype=np.uint8),
        }


class FakeRadarRow:
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.name = frame_id

    def __getitem__(self, key):
        if key == "frame_id":
            return self.frame_id
        raise KeyError(key)


class FakeSession:
    def __init__(self):
        self.heatmap_reads = []
        self.points_reads = []

    def load_radar_heatmap(self, row):
        self.heatmap_reads.append(row["frame_id"])
        return np.full((3, 3), row["frame_id"], dtype=np.float32)

    def load_radar_points(self, row):
        self.points_reads.append(row["frame_id"])
        return [{"x": float(row["frame_id"]), "y": 0.0, "z": 0.0, "velocity_mps": 0.0}]


def make_request(generation, svo=None, svo_index=None, session=None, radar_row=None, radar_delta_ns=None):
    return FrameRequest(
        generation=generation, svo=svo, svo_index=svo_index,
        session=session, radar_row=radar_row, radar_delta_ns=radar_delta_ns,
    )


# --------------------------------------------------------------------------- #
# LRUCache in isolation.
# --------------------------------------------------------------------------- #

def test_lru_cache_hit_and_miss_counts():
    cache = LRUCache(max_size=2)
    assert cache.get("a") is None
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_lru_cache_evicts_least_recently_used_beyond_max_size():
    cache = LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # "a" is least-recently-used -> evicted
    assert len(cache) == 2
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert cache.stats()["evictions"] == 1


def test_lru_cache_get_refreshes_recency():
    cache = LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")       # touch "a" so "b" becomes the least-recently-used
    cache.put("c", 3)    # evicts "b", not "a"
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_lru_cache_clear_resets_contents_but_not_counters():
    cache = LRUCache(max_size=2)
    cache.put("a", 1)
    cache.get("a")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None  # a fresh miss after clearing
    assert cache.stats()["hits"] == 1  # historical counters are diagnostics, not invalidated


def test_lru_cache_rejects_non_positive_size():
    with pytest.raises(ValueError):
        LRUCache(max_size=0)


# --------------------------------------------------------------------------- #
# Camera cache integration (requirement: keyed by exact SVO frame index).
# --------------------------------------------------------------------------- #

def test_camera_cache_hit_does_not_call_read_frame_again():
    worker = SensorWorker()
    svo = FakeSvo()
    results = []
    worker.cameraReady.connect(lambda gen, payload: results.append(payload))

    worker.process(make_request(1, svo=svo, svo_index=100))
    worker.process(make_request(2, svo=svo, svo_index=100))  # same frame index again

    assert svo.calls == [100]  # decoded only once
    assert len(results) == 2
    assert np.array_equal(results[0]["left"], results[1]["left"])


def test_camera_cache_repeated_access_pattern_100_101_100_101_100():
    worker = SensorWorker()
    svo = FakeSvo()
    worker.cameraReady.connect(lambda gen, payload: None)

    for i, idx in enumerate([100, 101, 100, 101, 100]):
        worker.process(make_request(i, svo=svo, svo_index=idx))

    # Each distinct index decoded exactly once; repeats served from cache.
    assert svo.calls == [100, 101]


def test_camera_cache_different_indices_are_not_conflated():
    worker = SensorWorker()
    svo = FakeSvo()
    results = []
    worker.cameraReady.connect(lambda gen, payload: results.append(payload))

    worker.process(make_request(1, svo=svo, svo_index=5))
    worker.process(make_request(2, svo=svo, svo_index=6))

    assert not np.array_equal(results[0]["left"], results[1]["left"])


def test_camera_cache_eviction_beyond_configured_limit():
    worker = SensorWorker(cache_config=CacheConfig(camera_max_size=2, radar_max_size=64))
    svo = FakeSvo()
    worker.cameraReady.connect(lambda gen, payload: None)

    for gen, idx in enumerate([1, 2, 3]):  # 3 distinct frames, cache holds only 2
        worker.process(make_request(gen, svo=svo, svo_index=idx))

    worker.process(make_request(99, svo=svo, svo_index=1))  # evicted -> re-decoded

    assert svo.calls == [1, 2, 3, 1]
    stats = worker.cache_stats()["camera"]
    assert stats["size"] <= 2
    assert stats["evictions"] >= 1


# --------------------------------------------------------------------------- #
# Radar cache integration (requirement: keyed by exact radar row/sample, and
# shared across camera frames that resolve to the same radar sample).
# --------------------------------------------------------------------------- #

def test_radar_cache_shared_across_two_camera_frames_same_radar_sample():
    worker = SensorWorker()
    session = FakeSession()
    row = FakeRadarRow(frame_id=50)

    # camera frame 100 -> radar sample 50, camera frame 101 -> radar sample 50
    worker.process(make_request(1, session=session, radar_row=row, radar_delta_ns=10))
    worker.process(make_request(2, session=session, radar_row=row, radar_delta_ns=-40))

    assert session.heatmap_reads == [50]
    assert session.points_reads == [50]


def test_radar_cache_delta_ns_reflects_current_request_not_the_cached_one():
    worker = SensorWorker()
    session = FakeSession()
    row = FakeRadarRow(frame_id=50)
    results = []
    worker.radarReady.connect(lambda gen, payload: results.append(payload))

    worker.process(make_request(1, session=session, radar_row=row, radar_delta_ns=10))
    worker.process(make_request(2, session=session, radar_row=row, radar_delta_ns=-40))

    assert session.heatmap_reads == [50]  # still only read once
    assert results[0]["delta_ns"] == 10
    assert results[1]["delta_ns"] == -40  # sync delta is per-request, not cached
    assert np.array_equal(results[0]["heatmap"], results[1]["heatmap"])  # same underlying data


def test_radar_cache_different_samples_not_conflated():
    worker = SensorWorker()
    session = FakeSession()

    worker.process(make_request(1, session=session, radar_row=FakeRadarRow(frame_id=50), radar_delta_ns=0))
    worker.process(make_request(2, session=session, radar_row=FakeRadarRow(frame_id=51), radar_delta_ns=0))

    assert session.heatmap_reads == [50, 51]


def test_radar_cache_eviction_beyond_configured_limit():
    worker = SensorWorker(cache_config=CacheConfig(camera_max_size=32, radar_max_size=2))
    session = FakeSession()

    for gen, fid in enumerate([1, 2, 3]):
        worker.process(make_request(gen, session=session, radar_row=FakeRadarRow(frame_id=fid), radar_delta_ns=0))
    worker.process(make_request(99, session=session, radar_row=FakeRadarRow(frame_id=1), radar_delta_ns=0))

    assert session.heatmap_reads == [1, 2, 3, 1]
    stats = worker.cache_stats()["radar"]
    assert stats["size"] <= 2
    assert stats["evictions"] >= 1


# --------------------------------------------------------------------------- #
# Cache hits must not bypass the UI-side generation check (worker itself is
# generation-agnostic: it just tags every emit with the request's own
# generation, cached or not -- the UI is what enforces staleness).
# --------------------------------------------------------------------------- #

def test_cache_hit_still_carries_the_requests_own_generation():
    worker = SensorWorker()
    svo = FakeSvo()
    seen = []
    worker.cameraReady.connect(lambda gen, payload: seen.append(gen))

    worker.process(make_request(1, svo=svo, svo_index=7))
    worker.process(make_request(2, svo=svo, svo_index=7))  # cache hit, different generation

    assert seen == [1, 2]  # each emit tagged with its own request's generation, not frozen at insert time


def test_stale_generation_with_cache_hit_is_still_rejected_by_ui(qtbot):
    """End-to-end: even when the worker serves a cache hit, MainWindow's
    existing generation check is what discards it if stale -- the cache
    changes nothing about that contract."""
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)

    win._generation = 5
    payload = {"left": np.zeros((2, 2, 3), dtype=np.uint8), "right": np.zeros((2, 2, 3), dtype=np.uint8)}
    win._on_camera_ready(4, payload)  # a hypothetical cache-hit result for an old generation

    pix = win.left_label.pixmap()
    assert pix is None or pix.isNull()
    win.close()


# --------------------------------------------------------------------------- #
# Session invalidation: caches must be cleared, and one session's identity
# must never let another session's cached entries be returned.
# --------------------------------------------------------------------------- #

def test_reset_caches_clears_both_camera_and_radar():
    worker = SensorWorker()
    svo = FakeSvo()
    session = FakeSession()

    worker.process(make_request(1, svo=svo, svo_index=1, session=session, radar_row=FakeRadarRow(1), radar_delta_ns=0))
    assert worker.cache_stats()["camera"]["size"] == 1
    assert worker.cache_stats()["radar"]["size"] == 1

    worker.reset_caches()

    assert worker.cache_stats()["camera"]["size"] == 0
    assert worker.cache_stats()["radar"]["size"] == 0


def test_session_a_frame_cannot_be_served_after_session_b_without_reset_being_bypassed():
    """Two different SvoReader/RecordingSession *instances* (as happens on a
    real session switch, where a brand new SvoReader/RecordingSession is
    constructed) must not collide in the cache even at the same index,
    because the cache key includes id(svo)/id(session)."""
    worker = SensorWorker()
    svo_a = FakeSvo(tag=1)
    svo_b = FakeSvo(tag=2)

    results = []
    worker.cameraReady.connect(lambda gen, payload: results.append(payload))

    worker.process(make_request(1, svo=svo_a, svo_index=0))
    worker.process(make_request(2, svo=svo_b, svo_index=0))  # same index, different session's SvoReader

    assert svo_a.calls == [0]
    assert svo_b.calls == [0]  # not served from svo_a's cache entry
    assert not np.array_equal(results[0]["left"], results[1]["left"])


def test_reset_caches_prevents_old_session_data_from_reappearing():
    worker = SensorWorker()
    svo_a = FakeSvo()
    results = []
    worker.cameraReady.connect(lambda gen, payload: results.append(payload))

    worker.process(make_request(1, svo=svo_a, svo_index=0))
    worker.reset_caches()  # simulates MainWindow's session-switch reset

    svo_b = FakeSvo()
    worker.process(make_request(2, svo=svo_b, svo_index=0))

    assert svo_b.calls == [0]  # forced a real decode, not a leftover cache entry


# --------------------------------------------------------------------------- #
# Existing behavior (errors, coalescing) must be unaffected by the cache.
# --------------------------------------------------------------------------- #

def test_worker_error_behavior_unchanged_with_cache_present():
    worker = SensorWorker()
    errors = []
    worker.errorOccurred.connect(lambda gen, kind, msg: errors.append((gen, kind, msg)))

    class RaisingSvo:
        def read_frame(self, index):
            raise RuntimeError("boom")

    worker.process(make_request(1, svo=RaisingSvo(), svo_index=0))

    assert errors == [(1, "camera", "boom")]
    assert worker.cache_stats()["camera"]["size"] == 0  # a failed decode is never cached


def test_coalescing_behavior_unchanged_with_cache_present(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)

    sent = []

    def fake_send(request):
        sent.append(request)

    win._send_to_worker = fake_send
    win._worker_busy = True

    req_a = FrameRequest(generation=1, svo=None, svo_index=None, session=None, radar_row=None, radar_delta_ns=None)
    req_b = FrameRequest(generation=2, svo=None, svo_index=None, session=None, radar_row=None, radar_delta_ns=None)
    win._pending_request = req_a
    win._pending_request = req_b

    win._on_worker_finished(1)

    assert win._worker_busy is False
    assert sent == [req_b]  # still coalesces to just the latest, cache or no cache
    win.close()
