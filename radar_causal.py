"""Display-only "causal" radar sample-selection policy.

Dataset Inspector's existing `SyncIndex` (see sync_index.py) answers "which
recorded radar sample is temporally CLOSEST to this camera frame?" -- and it
may legitimately pick a sample recorded slightly *after* the camera frame's
own timestamp, since it is a plain nearest-neighbor search. During normal
playback this can look like the radar view "seeing into the future" for an
instant, which is confusing even though nothing is actually wrong with the
recording or with SyncIndex.

This module adds exactly one additional, separate rule -- a presentation
policy, not a new synchronization system:

    SyncIndex (nearest)  : j = argmin_j |t_radar_j - t_camera|
    causal (this module) : j = max { j : t_radar_j <= t_camera }

i.e. "the latest radar sample that had already been recorded by the current
camera time." It is built from the exact same already-loaded timestamp
arrays SyncIndex itself is built from (`session.radar["timestamp_ns"]` and
`session.frames["timestamp_ns"]`), never touches disk, never mutates a
timestamp, and never replaces or feeds back into SyncIndex -- MainWindow
decides, per-frame, which of the two index arrays to look up, exactly the
same way it already looks up `SyncIndex.radar_for()`.
"""
from __future__ import annotations

import numpy as np

from sync_index import MISSING

__all__ = ["MISSING", "causal_indices_for_frames", "causal_index"]


def causal_indices_for_frames(sensor_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Vectorized causal lookup, one result per entry of `query_ts` (meant to
    be called once per session with the camera frames' timestamps, exactly
    like `sync_index._nearest_indices`, and the result stored/indexed by
    frame position the same way `SyncIndex.radar_index` is).

    Returns, for each query timestamp, the highest positional index `j` in
    `sensor_ts` (assumed sorted ascending, as `session.radar`/`session.frames`
    already are -- see session.py's `_read_parquet_dir`) such that
    `sensor_ts[j] <= query_ts`, or MISSING if no such index exists (including
    when `sensor_ts` is empty, matching SyncIndex's own empty-data
    convention).
    """
    query_ts = np.asarray(query_ts)
    n = len(sensor_ts)
    if n == 0 or query_ts.size == 0:
        return np.full(query_ts.shape, MISSING, dtype=np.int64)

    # side="right": the insertion point just past every sensor_ts entry
    # equal to a given query -- so subtracting 1 lands on the *last* sample
    # with sensor_ts <= query (an exact-timestamp match counts as eligible,
    # matching "t_radar <= t_camera").
    idx = np.searchsorted(sensor_ts, query_ts, side="right") - 1
    return np.where(idx < 0, MISSING, idx).astype(np.int64)


def causal_index(sensor_ts: np.ndarray, query_ts: int) -> int:
    """Scalar convenience wrapper around `causal_indices_for_frames` for a
    single timestamp -- used by tests; MainWindow itself uses the
    precomputed per-session array instead of calling this per-frame."""
    result = causal_indices_for_frames(np.asarray(sensor_ts), np.asarray([query_ts]))
    return int(result[0])
