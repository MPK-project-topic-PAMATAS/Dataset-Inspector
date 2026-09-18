"""Correctness tests for the precomputed nearest-timestamp index (see
sync_index.py), which must select exactly the same samples as the existing
scalar reference functions:

    RecordingSession.nearest_radar_row
    MtiData.nearest_index
    SvoReader.nearest_frame_index
    LidarReader.nearest_index

Nothing here writes to a recording directory, resamples, interpolates, or
changes what "nearest" means -- these tests only check that the vectorized,
precomputed version agrees with the scalar reference, one query at a time
and over an entire real session.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sync_index import MISSING, SyncIndex, _nearest_indices


def _reference_nearest(sensor_ts, t):
    """A direct transcription of the scalar rule shared by every existing
    nearest-timestamp function (RecordingSession.nearest_radar_row,
    MtiData.nearest_index, SvoReader.nearest_frame_index,
    LidarReader.nearest_index) -- used as the ground truth these tests check
    the vectorized implementation against. Returns None for empty input.
    """
    n = len(sensor_ts)
    if n == 0:
        return None
    idx = int(np.searchsorted(sensor_ts, t))
    if idx <= 0:
        return 0
    if idx >= n:
        return n - 1
    before, after = sensor_ts[idx - 1], sensor_ts[idx]
    return idx - 1 if (t - before) <= (after - t) else idx


# --------------------------------------------------------------------------- #
# _nearest_indices vs. the scalar reference, one case at a time.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("description,sensor_ts,query", [
    ("exact match on an interior sample", [0, 100, 200, 300], 200),
    ("exact match on the first sample", [0, 100, 200, 300], 0),
    ("exact match on the last sample", [0, 100, 200, 300], 300),
    ("between two samples, closer to the earlier one", [0, 100, 200, 300], 130),
    ("between two samples, closer to the later one", [0, 100, 200, 300], 180),
    ("before the first sample", [100, 200, 300], 10),
    ("after the last sample", [100, 200, 300], 999),
    ("equal-distance tie -- must pick the earlier sample", [90, 110], 100),
    ("irregularly spaced samples", [0, 5, 400, 405, 4000], 202),
    ("irregularly spaced, tie between far-apart samples", [0, 500], 250),
    ("single sample", [42], 999),
    ("duplicate timestamps, query matches the duplicated value", [10, 20, 20, 30], 20),
    ("duplicate timestamps, query falls before them", [10, 20, 20, 30], 15),
])
def test_nearest_indices_matches_scalar_reference(description, sensor_ts, query):
    sensor_ts = np.asarray(sensor_ts, dtype=np.int64)
    expected = _reference_nearest(sensor_ts, query)
    got = _nearest_indices(sensor_ts, np.asarray([query], dtype=np.int64))[0]
    assert got == expected, description


def test_nearest_indices_empty_sensor_returns_missing_for_every_query():
    result = _nearest_indices(np.array([], dtype=np.int64), np.asarray([1, 2, 3], dtype=np.int64))
    assert list(result) == [MISSING, MISSING, MISSING]


def test_nearest_indices_vectorized_matches_reference_across_many_random_queries():
    rng = np.random.default_rng(0)
    sensor_ts = np.sort(rng.integers(0, 1_000_000, size=200)).astype(np.int64)
    queries = rng.integers(-100_000, 1_100_000, size=500).astype(np.int64)

    got = _nearest_indices(sensor_ts, queries)
    expected = [_reference_nearest(sensor_ts, q) for q in queries]

    assert list(got) == expected


def test_nearest_indices_handles_duplicate_sensor_timestamps_like_reference():
    rng = np.random.default_rng(1)
    base = np.sort(rng.integers(0, 1000, size=50))
    sensor_ts = np.repeat(base, 2).astype(np.int64)  # every timestamp appears twice
    queries = rng.integers(-50, 1050, size=200).astype(np.int64)

    got = _nearest_indices(sensor_ts, queries)
    expected = [_reference_nearest(sensor_ts, q) for q in queries]

    assert list(got) == expected


# --------------------------------------------------------------------------- #
# SyncIndex against the *actual* existing methods, using tiny fakes shaped
# like RecordingSession/MtiData/LidarReader/SvoReader.
# --------------------------------------------------------------------------- #

class _FakeMti:
    def __init__(self, timestamp_ns):
        self.timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64)

    @property
    def n(self):
        return len(self.timestamp_ns)

    def nearest_index(self, timestamp_ns: int) -> int:  # the real reference implementation, copied verbatim
        idx = int(np.searchsorted(self.timestamp_ns, timestamp_ns))
        if idx <= 0:
            return 0
        if idx >= self.n:
            return self.n - 1
        before, after = self.timestamp_ns[idx - 1], self.timestamp_ns[idx]
        return idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx


class _FakeLidar:
    def __init__(self, timestamps_ns):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)

    @property
    def num_scans(self):
        return len(self.timestamps_ns)

    def nearest_index(self, timestamp_ns: int):  # the real reference implementation, copied verbatim
        if self.num_scans == 0:
            return None, None
        idx = int(np.searchsorted(self.timestamps_ns, timestamp_ns))
        if idx <= 0:
            idx = 0
        elif idx >= self.num_scans:
            idx = self.num_scans - 1
        else:
            before, after = self.timestamps_ns[idx - 1], self.timestamps_ns[idx]
            idx = idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx
        return idx, int(self.timestamps_ns[idx]) - int(timestamp_ns)


class _FakeSvo:
    def __init__(self, timestamps_ns):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self.num_frames = len(self.timestamps_ns)

    def nearest_frame_index(self, timestamp_ns: int) -> int:  # the real reference implementation, copied verbatim
        idx = int(np.searchsorted(self.timestamps_ns, timestamp_ns))
        if idx <= 0:
            return 0
        if idx >= self.num_frames:
            return self.num_frames - 1
        before, after = self.timestamps_ns[idx - 1], self.timestamps_ns[idx]
        return idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx


class _FakeSession:
    """Shaped like RecordingSession, but with tiny synthetic, irregular data."""

    def __init__(self, frame_ts, radar_ts, mti_ts, lidar_ts):
        import pandas as pd

        self.session_id = "fake"
        self.frames = pd.DataFrame({"timestamp_ns": np.asarray(frame_ts, dtype=np.int64)})
        self.radar = (
            pd.DataFrame({"timestamp_ns": np.asarray(radar_ts, dtype=np.int64)})
            if len(radar_ts) else pd.DataFrame()
        )
        self.mti = _FakeMti(mti_ts) if len(mti_ts) else None
        self.lidar = _FakeLidar(lidar_ts) if len(lidar_ts) else None

    def nearest_radar_row(self, timestamp_ns: int):  # the real reference implementation, copied verbatim
        if self.radar.empty:
            return None, None
        ts = self.radar["timestamp_ns"].to_numpy()
        idx = int(np.searchsorted(ts, timestamp_ns))
        if idx <= 0:
            idx = 0
        elif idx >= len(ts):
            idx = len(ts) - 1
        else:
            before, after = ts[idx - 1], ts[idx]
            idx = idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx
        row = self.radar.iloc[idx]
        return row, int(row["timestamp_ns"]) - int(timestamp_ns)


def _make_synthetic_session():
    rng = np.random.default_rng(42)
    frame_ts = np.sort(rng.integers(0, 1_000_000, size=60)).astype(np.int64)
    radar_ts = np.sort(rng.integers(0, 1_000_000, size=25)).astype(np.int64)
    mti_ts = np.sort(rng.integers(0, 1_000_000, size=400)).astype(np.int64)
    lidar_ts = np.sort(rng.integers(0, 1_000_000, size=18)).astype(np.int64)
    return _FakeSession(frame_ts, radar_ts, mti_ts, lidar_ts)


def test_sync_index_matches_every_reference_function_over_a_synthetic_session():
    session = _make_synthetic_session()
    svo = _FakeSvo(np.sort(np.random.default_rng(7).integers(0, 1_000_000, size=60)))

    index = SyncIndex.build(session, svo)

    frame_ts = session.frames["timestamp_ns"].to_numpy()
    for i, ts in enumerate(frame_ts):
        expected_row, _ = session.nearest_radar_row(int(ts))
        # nearest_radar_row returns a *row*; compare its content against
        # the row SyncIndex's precomputed position points at, rather than
        # re-deriving a position, to stay a faithful ground truth
        # independent of SyncIndex's own internals.
        got_pos = index.radar_for(i)
        got_row = session.radar.iloc[got_pos]
        assert int(got_row["timestamp_ns"]) == int(expected_row["timestamp_ns"])

        assert index.mti_for(i) == session.mti.nearest_index(int(ts))

        expected_lidar_idx, _ = session.lidar.nearest_index(int(ts))
        assert index.lidar_for(i) == expected_lidar_idx

        assert index.svo_for(i) == svo.nearest_frame_index(int(ts))


def test_sync_index_missing_sensor_represented_as_missing_not_fabricated():
    session = _FakeSession(
        frame_ts=[0, 100, 200],
        radar_ts=[],   # no radar data at all
        mti_ts=[10, 20],
        lidar_ts=[],   # no lidar data at all
    )
    index = SyncIndex.build(session, svo=None)

    assert all(index.radar_for(i) == MISSING for i in range(3))
    assert all(index.lidar_for(i) == MISSING for i in range(3))
    assert all(index.svo_for(i) == MISSING for i in range(3))  # no svo given
    assert all(index.mti_for(i) != MISSING for i in range(3))  # mti *is* present


def test_sync_index_stats_report_expected_counts():
    session = _FakeSession(frame_ts=[0, 100, 200, 300], radar_ts=[50], mti_ts=[], lidar_ts=[10, 20])
    index = SyncIndex.build(session, svo=None)

    stats = index.stats()
    assert stats["camera_frames"] == 4
    assert stats["radar_mappings"] == 4
    assert stats["radar_unavailable"] == 0
    assert stats["mti_mappings"] == 0
    assert stats["mti_unavailable"] == 4
    assert stats["lidar_mappings"] == 4


def test_sync_index_matches_uses_identity_not_frame_count():
    session_a = _make_synthetic_session()
    session_b = _make_synthetic_session()  # same shape/size, different instance

    index_a = SyncIndex.build(session_a, svo=None)

    assert index_a.matches(session_a)
    assert not index_a.matches(session_b)  # same frame count, but not the same session


# --------------------------------------------------------------------------- #
# Full-session regression test against the real recorded dataset, if present.
# --------------------------------------------------------------------------- #

_DATASET_ROOT = Path(__file__).parent.parent / "output_multi-record" / "output"


def _first_real_session_dir():
    if not _DATASET_ROOT.exists():
        return None
    for p in sorted(_DATASET_ROOT.iterdir()):
        if p.is_dir() and (p / "raw" / "recording.svo2").exists():
            return p
    return None


@pytest.mark.skipif(_first_real_session_dir() is None, reason="no real recorded session available")
def test_sync_index_matches_reference_for_every_frame_of_a_real_session():
    pytest.importorskip("pyzed.sl")
    from session import RecordingSession
    from svo_reader import SvoReader

    session_dir = _first_real_session_dir()
    session = RecordingSession.load(session_dir)
    svo = SvoReader(session.svo_path)

    index = SyncIndex.build(session, svo)
    frame_ts = session.frames["timestamp_ns"].to_numpy()
    assert index.num_frames == len(frame_ts) > 0

    for i, ts in enumerate(frame_ts):
        ts = int(ts)

        expected_row, _ = session.nearest_radar_row(ts)
        got_radar_pos = index.radar_for(i)
        if expected_row is None:
            assert got_radar_pos == MISSING
        else:
            got_row = session.radar.iloc[got_radar_pos]
            assert int(got_row["timestamp_ns"]) == int(expected_row["timestamp_ns"]), f"radar mismatch at frame {i}"

        if session.mti is not None and session.mti.n > 0:
            assert index.mti_for(i) == session.mti.nearest_index(ts), f"mti mismatch at frame {i}"
        else:
            assert index.mti_for(i) == MISSING

        if session.lidar is not None and session.lidar.num_scans > 0:
            expected_lidar_idx, _ = session.lidar.nearest_index(ts)
            assert index.lidar_for(i) == expected_lidar_idx, f"lidar mismatch at frame {i}"
        else:
            assert index.lidar_for(i) == MISSING

        assert index.svo_for(i) == svo.nearest_frame_index(ts), f"svo mismatch at frame {i}"

    svo.close()
