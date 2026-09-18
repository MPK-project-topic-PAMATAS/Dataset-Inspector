"""Background worker for Dataset Inspector's blocking per-frame I/O.

Dataset Inspector is a read-only viewer. This worker never writes to the
recording directory, never resamples, interpolates, or re-synchronizes
anything, and never decides *which* frame/timestamp to fetch. The UI
thread's existing nearest-timestamp logic (RecordingSession.nearest_radar_row,
SvoReader.nearest_frame_index) picks what to fetch; this worker only performs
the resulting blocking operations -- SVO decode and radar file reads -- off
the Qt UI thread.

It lives in a single dedicated QThread (see MainWindow) and processes one
request at a time. Results go back to the UI thread purely through Qt
signals, which Qt automatically delivers via a queued connection across the
thread boundary, so no widget is ever touched from this thread.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, Slot

from cache import CacheConfig, LRUCache

logger = logging.getLogger(__name__)

# Log accumulated cache hit/miss/eviction counters this often, so playback at
# 30 FPS doesn't flood the console with a line every frame.
_STATS_LOG_INTERVAL = 100


class FrameRequest:
    """Everything the worker needs for one frame.

    Every field is a read-only reference or value already resolved by the
    UI thread's existing synchronization logic -- the worker performs no
    timestamp math of its own.
    """

    __slots__ = ("generation", "svo", "svo_index", "session", "radar_row", "radar_delta_ns")

    def __init__(self, generation, svo, svo_index, session, radar_row, radar_delta_ns):
        self.generation = generation
        self.svo = svo
        self.svo_index = svo_index
        self.session = session
        self.radar_row = radar_row
        self.radar_delta_ns = radar_delta_ns


class SensorWorker(QObject):
    """Lives in its own QThread. Processes one request at a time.

    Adds a small bounded cache in front of the two blocking operations
    (SVO decode, radar file reads) so that re-visiting a frame or radar
    sample already seen recently -- the common case while scrubbing back
    and forth -- doesn't re-decode/re-read it from the recording. This is
    a pure speed-up: a cache hit returns exactly the same data a fresh
    decode/read would have produced, and it never bypasses the generation
    check on the UI side (the cached payload is still emitted tagged with
    the request's own generation, exactly like a fresh result).

    Cache ownership: `_camera_cache`/`_radar_cache` are created here and
    only ever touched from `process()`/`reset_caches()`, both of which run
    exclusively on this worker's own thread. Nothing on the UI thread reads
    or writes them directly, so no locking is used (see cache.LRUCache).
    """

    cameraReady = Signal(int, object)      # generation, {"left": ndarray, "right": ndarray}
    radarReady = Signal(int, object)       # generation, {"heatmap": ndarray|None, "points": list, "delta_ns": int|None}
    errorOccurred = Signal(int, str, str)  # generation, kind ("camera"|"radar"), message
    finished = Signal(int)                 # generation; always emitted last for a request

    def __init__(self, cache_config: CacheConfig | None = None, parent=None):
        super().__init__(parent)
        config = cache_config or CacheConfig()
        self._camera_cache = LRUCache(config.camera_max_size, name="camera_cache")
        self._radar_cache = LRUCache(config.radar_max_size, name="radar_cache")
        self._processed_count = 0

    @Slot()
    def reset_caches(self) -> None:
        """Drop all cached frames/radar samples.

        Called on session switch so a frame decoded from session A's SVO2
        (or a radar sample read from session A's files) can never be
        returned once session B is active -- see MainWindow._load_session.
        """
        self._camera_cache.clear()
        self._radar_cache.clear()

    def cache_stats(self) -> dict:
        """Diagnostic snapshot; only ever called from this worker's own
        thread (e.g. from process() below), never from the UI thread."""
        return {"camera": self._camera_cache.stats(), "radar": self._radar_cache.stats()}

    @Slot(object)
    def process(self, request: FrameRequest) -> None:
        gen = request.generation

        if request.svo is not None and request.svo_index is not None:
            self._process_camera(gen, request)

        if request.session is not None and request.radar_row is not None:
            self._process_radar(gen, request)

        self._processed_count += 1
        if self._processed_count % _STATS_LOG_INTERVAL == 0:
            logger.debug("cache stats: %s", self.cache_stats())

        self.finished.emit(gen)

    def _process_camera(self, gen: int, request: FrameRequest) -> None:
        # Identifies the exact decoded SVO frame, not a timestamp: the SVO
        # reader itself works in frame indices (see SvoReader.read_frame),
        # and a session switch gets a brand new SvoReader instance, so
        # id(svo) alone already scopes the key to the current session.
        cache_key = (id(request.svo), request.svo_index)

        cached = self._camera_cache.get(cache_key)
        if cached is not None:
            self.cameraReady.emit(gen, cached)
            return

        try:
            frame = request.svo.read_frame(request.svo_index)
        except Exception as e:
            logger.warning("camera decode failed (gen=%s): %s", gen, e)
            self.errorOccurred.emit(gen, "camera", str(e))
            return

        if not frame.get("ok"):
            self.errorOccurred.emit(gen, "camera", "frame decode failed")
            return

        # frame["left"]/["right"] are already freshly-allocated arrays (see
        # SvoReader._to_rgb's .copy(), needed because the ZED SDK reuses its
        # internal buffer on the next grab()) -- storing them as-is in the
        # cache adds no further copy, since nothing downstream mutates them.
        payload = {"left": frame["left"], "right": frame["right"]}
        self._camera_cache.put(cache_key, payload)
        self.cameraReady.emit(gen, payload)

    def _process_radar(self, gen: int, request: FrameRequest) -> None:
        # Identifies the exact recorded radar sample (its own frame_id, or
        # failing that the row's stable positional label), not a timestamp
        # or the requesting camera frame -- so two camera frames that
        # resolve to the same nearest radar row share one cache entry.
        cache_key = (id(request.session), self._radar_row_key(request.radar_row))

        cached = self._radar_cache.get(cache_key)
        if cached is not None:
            # delta_ns is how far *this* camera frame's timestamp is from
            # the radar sample's timestamp -- it depends on the request,
            # not the radar sample itself, so it rides along fresh on every
            # emit rather than being frozen into the cached entry.
            payload = {**cached, "delta_ns": request.radar_delta_ns}
            self.radarReady.emit(gen, payload)
            return

        try:
            heatmap = request.session.load_radar_heatmap(request.radar_row)
            points = request.session.load_radar_points(request.radar_row)
        except Exception as e:
            logger.warning("radar load failed (gen=%s): %s", gen, e)
            self.errorOccurred.emit(gen, "radar", str(e))
            return

        cached_payload = {"heatmap": heatmap, "points": points}
        self._radar_cache.put(cache_key, cached_payload)
        self.radarReady.emit(gen, {**cached_payload, "delta_ns": request.radar_delta_ns})

    @staticmethod
    def _radar_row_key(row):
        try:
            return int(row["frame_id"])
        except Exception:
            return int(row.name)
