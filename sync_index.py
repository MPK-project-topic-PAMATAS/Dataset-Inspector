"""Precomputed nearest-timestamp lookup index, built once per session.

Dataset Inspector's existing nearest-neighbor functions --
`RecordingSession.nearest_radar_row`, `MtiData.nearest_index`,
`SvoReader.nearest_frame_index`, `LidarReader.nearest_index` -- are each
called with the *same* camera-frame timestamp on every render (and, for the
SVO index, from two different call sites with the same timestamp -- the
camera decode request and the ZED IMU cursor). This module computes, once,
the same `argmin_j |t_sensor[j] - t_camera[i]|` result those functions
already produce for every camera frame `i`, and keeps only the resulting
integer indices -- never a copy of the underlying sensor data, and never a
new tie-break or boundary rule.

This is read-only bookkeeping: it changes nothing about which recorded
sample gets selected, only how many times the search for it is repeated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for "no valid sample exists for this camera frame", matching the
# existing None-returning / early-exit behavior of the reference functions
# when a sensor's data is empty or absent -- never a fabricated sample.
MISSING = -1


def _nearest_indices(sensor_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Vectorized form of the exact scalar rule shared by every existing
    nearest-timestamp function in this codebase:

        idx = searchsorted(sensor_ts, t, side="left")
        if idx <= 0: idx = 0
        elif idx >= len(sensor_ts): idx = len(sensor_ts) - 1
        else: idx = idx - 1 if (t - sensor_ts[idx-1]) <= (sensor_ts[idx] - t) else idx

    i.e. argmin_j |sensor_ts[j] - t|, ties broken toward the earlier (lower)
    index. This is precisely what RecordingSession.nearest_radar_row,
    MtiData.nearest_index, SvoReader.nearest_frame_index and
    LidarReader.nearest_index each do, one query at a time; this function
    only vectorizes it across every camera-frame timestamp at once.

    Returns MISSING for every query when sensor_ts is empty, matching those
    functions' empty-data behavior (they return None/skip rather than
    inventing a sample).
    """
    query_ts = np.asarray(query_ts)
    n = len(sensor_ts)
    if n == 0 or query_ts.size == 0:
        return np.full(query_ts.shape, MISSING, dtype=np.int64)

    idx = np.searchsorted(sensor_ts, query_ts, side="left")
    idx_after = np.clip(idx, 0, n - 1)   # matches "idx >= n -> n-1" and the in-range "after" candidate
    idx_before = np.clip(idx - 1, 0, n - 1)  # matches the in-range "before" candidate

    diff_before = query_ts - sensor_ts[idx_before]
    diff_after = sensor_ts[idx_after] - query_ts

    # Only compare before-vs-after when idx is strictly inside (0, n); the
    # idx<=0 and idx>=n boundary cases just take idx_after (which clip()
    # already pinned to 0 or n-1), exactly like the scalar reference.
    use_before = (idx > 0) & (idx < n) & (diff_before <= diff_after)
    return np.where(use_before, idx_before, idx_after).astype(np.int64)


@dataclass
class SyncIndex:
    """One instance per (RecordingSession, SvoReader) pair; read-only once built.

    Each array is indexed by camera-frame position (0..num_frames-1, the
    same positional order as `RecordingSession.frames`) and holds the
    sensor sample index the *existing* nearest-timestamp function would
    have selected for that frame's timestamp, or MISSING if that sensor has
    no usable data at all -- matching the existing empty/absent-sensor
    behavior exactly.
    """

    session_identity: int          # id(session) this index was built from
    svo_identity: int | None       # id(svo) this index was built from, or None if no camera
    session_id: str                # human-readable session name, for logging only
    num_frames: int
    radar_index: np.ndarray        # camera frame i -> positional row in session.radar
    lidar_index: np.ndarray        # camera frame i -> scan index in LidarReader
    mti_index: np.ndarray          # camera frame i -> sample index in MtiData
    svo_index: np.ndarray          # camera frame i -> internal SVO frame index

    @classmethod
    def build(cls, session, svo=None) -> "SyncIndex":
        frames = session.frames
        frames_ts = frames["timestamp_ns"].to_numpy() if not frames.empty else np.array([], dtype=np.int64)
        num_frames = len(frames_ts)

        radar_ts = (
            session.radar["timestamp_ns"].to_numpy()
            if session.radar is not None and not session.radar.empty
            else np.array([], dtype=np.int64)
        )
        radar_index = _nearest_indices(radar_ts, frames_ts)

        if session.lidar is not None and session.lidar.num_scans > 0:
            lidar_index = _nearest_indices(session.lidar.timestamps_ns, frames_ts)
        else:
            lidar_index = np.full(num_frames, MISSING, dtype=np.int64)

        if session.mti is not None and session.mti.n > 0:
            mti_index = _nearest_indices(session.mti.timestamp_ns, frames_ts)
        else:
            mti_index = np.full(num_frames, MISSING, dtype=np.int64)

        if svo is not None:
            svo_index = _nearest_indices(svo.timestamps_ns, frames_ts)
        else:
            svo_index = np.full(num_frames, MISSING, dtype=np.int64)

        index = cls(
            session_identity=id(session),
            svo_identity=id(svo) if svo is not None else None,
            session_id=session.session_id,
            num_frames=num_frames,
            radar_index=radar_index,
            lidar_index=lidar_index,
            mti_index=mti_index,
            svo_index=svo_index,
        )
        logger.info("SyncIndex built for session=%s: %s", index.session_id, index.stats())
        return index

    def matches(self, session, svo=None) -> bool:
        """True only if this index was built from exactly this RecordingSession
        instance and (when one is given) this exact SvoReader instance.

        Deliberately identity-based (id()), not session_id or frame count:
        two different RecordingSession objects for the same recording (e.g.
        a session reloaded) must not be considered interchangeable, and a
        differently-sized SyncIndex from a *different* recording that
        happens to have the same frame count must never be mistaken for a
        match either.
        """
        return self.session_identity == id(session) and self.svo_identity == (id(svo) if svo is not None else None)

    # -- lookups, O(1) array reads, no search --------------------------- #
    def radar_for(self, frame_index: int) -> int:
        return int(self.radar_index[frame_index])

    def lidar_for(self, frame_index: int) -> int:
        return int(self.lidar_index[frame_index])

    def mti_for(self, frame_index: int) -> int:
        return int(self.mti_index[frame_index])

    def svo_for(self, frame_index: int) -> int:
        return int(self.svo_index[frame_index])

    def stats(self) -> dict:
        def summarize(name: str, arr: np.ndarray) -> dict:
            unavailable = int(np.count_nonzero(arr == MISSING))
            return {f"{name}_mappings": int(arr.size - unavailable), f"{name}_unavailable": unavailable}

        out = {"camera_frames": self.num_frames}
        out.update(summarize("radar", self.radar_index))
        out.update(summarize("lidar", self.lidar_index))
        out.update(summarize("mti", self.mti_index))
        out.update(summarize("svo", self.svo_index))
        return out
