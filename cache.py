"""A small, bounded LRU cache used by SensorWorker to avoid re-decoding the
same SVO camera frame or re-reading the same radar heatmap/points files
during playback and scrubbing.

This exists purely as an in-memory speed-up. It never changes what data is
displayed: a cache hit returns exactly the same decoded frame / loaded radar
sample a cache miss would have produced by reading the recording again. The
recorded dataset on disk is never read from or written to by this module.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable

logger = logging.getLogger(__name__)

# Sensible, easy-to-change defaults. Camera frames are decoded RGB images
# (a 1280x720 left+right pair is ~5.5MB), so a modest size keeps memory use
# reasonable while still covering the common "scrub back and forth over the
# last second or two" pattern. Radar entries are tiny (a small heatmap array
# plus a short list of detections), so a larger size costs little.
DEFAULT_CAMERA_CACHE_SIZE = 32
DEFAULT_RADAR_CACHE_SIZE = 64


@dataclass
class CacheConfig:
    """Bundles the two cache limits so callers can override both at once."""

    camera_max_size: int = DEFAULT_CAMERA_CACHE_SIZE
    radar_max_size: int = DEFAULT_RADAR_CACHE_SIZE


class LRUCache:
    """A tiny, dependency-free bounded LRU cache.

    Ownership note: instances of this class used by SensorWorker are created
    and accessed exclusively from within the worker thread (SensorWorker's
    slots all run there, and nothing on the UI thread ever touches these
    dicts directly), so no locking is added here. Introducing locks would
    just add overhead for a cache nothing else can see.
    """

    def __init__(self, max_size: int, name: str = "cache"):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.name = name
        self._data: "OrderedDict[Hashable, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Hashable) -> Any | None:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: Hashable, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        cleared = len(self._data)
        self._data.clear()
        if cleared:
            logger.debug("%s: cleared %d entries", self.name, cleared)

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._data),
            "max_size": self.max_size,
            "evictions": self.evictions,
        }
