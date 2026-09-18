"""Tests for radar_causal.py's display-only CAUSAL selection rule.

Pure-function tests, no Qt/session/SyncIndex involved -- see
test_main_window_radar_view.py for the MainWindow-level integration
(2D/3D toggle, NEAREST-vs-CAUSAL dispatch, the future-sample indicator).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_causal import MISSING, causal_index, causal_indices_for_frames  # noqa: E402


# --------------------------------------------------------------------------- #
# The exact worked example from the design brief.
# --------------------------------------------------------------------------- #

CAMERA_TS = np.array([100, 200, 300], dtype=np.int64)
RADAR_TS = np.array([90, 150, 220, 310], dtype=np.int64)


def test_camera_200_causal_is_150_nearest_would_be_220():
    # NEAREST (not this module's job, shown only for contrast): |90-200|=110,
    # |150-200|=50, |220-200|=20, |310-200|=110 -> nearest is 220 (index 2).
    # CAUSAL: eligible = {90, 150} -> latest = 150 (index 1).
    assert causal_index(RADAR_TS, 200) == 1
    assert RADAR_TS[causal_index(RADAR_TS, 200)] == 150


def test_camera_100_exact_timestamp_match_is_causally_eligible():
    assert causal_index(RADAR_TS, 100) == 0
    assert RADAR_TS[causal_index(RADAR_TS, 100)] == 90


def test_camera_80_before_first_radar_sample_has_no_causal_sample():
    """Mandatory case from the design brief: NEAREST would pick 90ms, but
    CAUSAL must NEVER fall back to a future sample -- it must report
    "no sample" instead."""
    assert causal_index(RADAR_TS, 80) == MISSING


def test_vectorized_matches_scalar_for_the_full_worked_example():
    result = causal_indices_for_frames(RADAR_TS, CAMERA_TS)
    expected = [causal_index(RADAR_TS, t) for t in CAMERA_TS]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, [0, 1, 2])  # 90, 150, 220


# --------------------------------------------------------------------------- #
# Boundary / edge cases (design brief §21).
# --------------------------------------------------------------------------- #

def test_exact_timestamp_at_last_radar_sample():
    assert causal_index(RADAR_TS, 310) == 3


def test_after_final_radar_sample_selects_the_last_one():
    assert causal_index(RADAR_TS, 10_000) == 3


def test_before_first_radar_sample_by_a_lot():
    assert causal_index(RADAR_TS, -1) == MISSING


def test_empty_radar_timestamps_always_missing():
    empty = np.array([], dtype=np.int64)
    assert causal_index(empty, 100) == MISSING
    result = causal_indices_for_frames(empty, CAMERA_TS)
    assert np.all(result == MISSING)


def test_irregular_radar_timestamps():
    ts = np.array([0, 5, 400, 401, 1000], dtype=np.int64)
    assert causal_index(ts, 6) == 1        # 5
    assert causal_index(ts, 399) == 1      # still 5, 400 not yet eligible
    assert causal_index(ts, 400) == 2      # 400
    assert causal_index(ts, 999) == 3      # 401
    assert causal_index(ts, 1000) == 4     # 1000


def test_duplicate_radar_timestamps_selects_last_duplicate():
    ts = np.array([0, 100, 100, 100, 200], dtype=np.int64)
    idx = causal_index(ts, 100)
    assert idx == 3
    assert ts[idx] == 100


def test_large_timestamp_gap_does_not_confuse_selection():
    ts = np.array([0, 1_000_000_000_000], dtype=np.int64)  # ~1000s gap
    assert causal_index(ts, 500_000_000_000) == 0
    assert causal_index(ts, 1_000_000_000_000) == 1


def test_single_radar_sample():
    ts = np.array([500], dtype=np.int64)
    assert causal_index(ts, 499) == MISSING
    assert causal_index(ts, 500) == 0
    assert causal_index(ts, 501) == 0


# --------------------------------------------------------------------------- #
# Simulated rapid camera advancement / reverse scrubbing / playback speeds --
# the causal invariant (displayed <= camera) must hold at every query.
# --------------------------------------------------------------------------- #

def test_rapid_forward_and_reverse_scrubbing_never_selects_a_future_sample():
    rng = np.random.default_rng(0)
    radar_ts = np.sort(rng.integers(0, 100_000, size=200)).astype(np.int64)
    camera_ts = rng.integers(-1000, 101_000, size=500).astype(np.int64)  # includes before/after range
    idx = causal_indices_for_frames(radar_ts, camera_ts)
    for t, j in zip(camera_ts, idx):
        if j == MISSING:
            assert not np.any(radar_ts <= t)
        else:
            assert radar_ts[j] <= t
            # must be the LATEST eligible sample, not merely some eligible one
            assert not np.any((radar_ts > radar_ts[j]) & (radar_ts <= t))


def test_first_and_final_camera_frame():
    # First camera frame (before any radar sample -> no causal sample).
    assert causal_index(RADAR_TS, CAMERA_TS[0] - 50) == MISSING
    # Final camera frame, well after the last radar sample.
    assert causal_index(RADAR_TS, RADAR_TS[-1] + 1000) == 3
