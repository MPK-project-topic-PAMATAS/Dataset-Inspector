"""Tests for the read-only Timeline Diagnostics widget (timeline_diagnostics.py)
and its wiring into MainWindow.

The widget itself never touches RecordingSession/SyncIndex/SvoReader -- it
only receives plain data (coverage ranges, a SensorHealthSnapshot, recorded
drop events) via `set_session`/`set_current`. These tests build small real
`SensorDiagnostic`/`SensorHealthSnapshot`/`TimelineEvent` instances (from
sensor_health.py) rather than reimplementing them, to prove the widget is a
pure consumer of values already computed elsewhere.
"""
from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensor_health import SensorDiagnostic, SensorHealthSnapshot, SensorState, TimelineEvent
from timeline_diagnostics import TimelineDiagnostics


def _snapshot(camera_ts=1_000_000_000, frame_index=0, radar=None, lidar=None, mti=None, zed_imu=None, gnss=None):
    def default(name):
        return SensorDiagnostic(name=name, state=SensorState.NO_DATA)

    return SensorHealthSnapshot(
        camera_timestamp_ns=camera_ts,
        camera_frame_index=frame_index,
        camera=SensorDiagnostic("Camera", SensorState.AVAILABLE, sample_index=frame_index,
                                 sensor_timestamp_ns=camera_ts, delta_ns=0, sync_class="GOOD"),
        radar=radar or default("Radar"),
        lidar=lidar or default("Lidar"),
        mti=mti or default("MTi-8"),
        zed_imu=zed_imu or default("ZED IMU"),
        gnss=gnss or default("GNSS"),
    )


@pytest.fixture
def widget(qtbot):
    w = TimelineDiagnostics()
    qtbot.addWidget(w)
    return w


def _force_layout(widget) -> None:
    """Forces _lane_rects to be computed deterministically, without relying
    on the offscreen QPA platform actually delivering a real paintEvent."""
    from PySide6.QtGui import QPainter, QPixmap

    pixmap = QPixmap(max(widget.width(), 1), max(widget.height(), 1))
    painter = QPainter(pixmap)
    try:
        widget._paint(painter)
    finally:
        painter.end()


# --------------------------------------------------------------------------- #
# Camera axis / coverage from recorded data.
# --------------------------------------------------------------------------- #

def test_coverage_matches_recorded_data_exactly(widget):
    coverage = {
        "Camera": (1_000_000_000, 2_000_000_000),
        "Radar": (1_010_000_000, 1_990_000_000),
        "Lidar": (990_000_000, 2_010_000_000),
    }
    widget.set_session(1_000_000_000, 2_000_000_000, coverage, [])

    assert widget.coverage() == coverage  # exactly what was recorded, not inferred/adjusted


def test_camera_timeline_span_uses_existing_camera_timestamps(widget):
    widget.set_session(5_000_000_000, 6_000_000_000, {"Camera": (5_000_000_000, 6_000_000_000)}, [])
    assert widget._fraction(5_500_000_000) == pytest.approx(0.5)
    assert widget._fraction(5_000_000_000) == pytest.approx(0.0)
    assert widget._fraction(6_000_000_000) == pytest.approx(1.0)


def test_missing_sensor_produces_no_fabricated_coverage(widget):
    # No "Lidar" key at all -- not a zero-width bar, not an estimated range.
    coverage = {"Camera": (0, 1_000_000_000), "Radar": (0, 1_000_000_000)}
    widget.set_session(0, 1_000_000_000, coverage, [])
    assert "Lidar" not in widget.coverage()
    assert "MTi-8" not in widget.coverage()
    assert "GNSS" not in widget.coverage()


# --------------------------------------------------------------------------- #
# Current cursor / selected samples / synchronization delta all come
# straight from the SensorHealthSnapshot -- nothing is recomputed.
# --------------------------------------------------------------------------- #

def test_current_cursor_corresponds_to_current_camera_frame(widget):
    widget.set_session(0, 1_000_000_000, {"Camera": (0, 1_000_000_000)}, [])
    snap = _snapshot(camera_ts=250_000_000, frame_index=7)
    widget.set_current(snap)

    assert widget._diag_for_lane("Camera").sensor_timestamp_ns == 250_000_000
    assert widget._diag_for_lane("Camera").sample_index == 7


def test_radar_selected_sample_comes_from_the_snapshot_sync_index_field(widget):
    radar_diag = SensorDiagnostic("Radar", SensorState.AVAILABLE, sample_index=453,
                                   sensor_timestamp_ns=1_005_000_000, delta_ns=5_000_000, sync_class="GOOD")
    snap = _snapshot(radar=radar_diag)
    widget.set_current(snap)

    got = widget._diag_for_lane("Radar")
    assert got is radar_diag  # the exact object SensorHealthModel produced, not a new calculation
    assert got.sample_index == 453
    assert got.sensor_timestamp_ns == 1_005_000_000


def test_lidar_selected_sample_comes_from_the_snapshot_sync_index_field(widget):
    lidar_diag = SensorDiagnostic("Lidar", SensorState.AVAILABLE, sample_index=12,
                                   sensor_timestamp_ns=999_000_000, delta_ns=-1_000_000, sync_class="GOOD")
    snap = _snapshot(lidar=lidar_diag)
    widget.set_current(snap)
    assert widget._diag_for_lane("Lidar") is lidar_diag


def test_mti_selected_sample_comes_from_the_snapshot_sync_index_field(widget):
    mti_diag = SensorDiagnostic("MTi-8", SensorState.AVAILABLE, sample_index=1900,
                                 sensor_timestamp_ns=1_000_800_000, delta_ns=800_000, sync_class="GOOD")
    snap = _snapshot(mti=mti_diag)
    widget.set_current(snap)
    assert widget._diag_for_lane("MTi-8") is mti_diag


def test_synchronization_delta_matches_sensor_health_values(widget):
    radar_diag = SensorDiagnostic("Radar", SensorState.STALE, sample_index=453,
                                   sensor_timestamp_ns=1_093_519_006, delta_ns=93_519_006, sync_class="STALE")
    snap = _snapshot(radar=radar_diag)
    widget.set_current(snap)

    got = widget._diag_for_lane("Radar")
    assert got.delta_ns == 93_519_006
    assert got.delta_ms == pytest.approx(93.519006)
    assert got.state == SensorState.STALE  # display classification, not recalculated here


def test_threshold_driven_state_changes_do_not_alter_selected_sample_or_delta(widget):
    """Two diagnostics for the exact same sample/delta but different
    threshold-driven states (as SyncThresholds would produce) must not
    change what the widget considers the selected sample or delta -- only
    the display state/color."""
    lenient = SensorDiagnostic("Radar", SensorState.AVAILABLE, sample_index=453,
                                sensor_timestamp_ns=1_050_000_000, delta_ns=50_000_000, sync_class="GOOD")
    strict = SensorDiagnostic("Radar", SensorState.STALE, sample_index=453,
                               sensor_timestamp_ns=1_050_000_000, delta_ns=50_000_000, sync_class="STALE")

    widget.set_current(_snapshot(radar=lenient))
    a = widget._diag_for_lane("Radar")
    widget.set_current(_snapshot(radar=strict))
    b = widget._diag_for_lane("Radar")

    assert a.sample_index == b.sample_index == 453
    assert a.sensor_timestamp_ns == b.sensor_timestamp_ns == 1_050_000_000
    assert a.delta_ns == b.delta_ns == 50_000_000
    assert a.state != b.state  # only the classification differs


# --------------------------------------------------------------------------- #
# Recorded events: timestamps/details preserved verbatim, click seeks via
# the existing shared path.
# --------------------------------------------------------------------------- #

def test_event_marker_timestamps_match_recorded_event_timestamps(widget):
    events = [
        TimelineEvent(event_type="RADAR_DROPPED_FRAMES", description="dropped=15", timestamp_ns=1_500_000_000),
        TimelineEvent(event_type="LIDAR_DROPPED_FRAMES", description="dropped=41", timestamp_ns=1_800_000_000),
    ]
    widget.set_session(1_000_000_000, 2_000_000_000, {"Camera": (1_000_000_000, 2_000_000_000)}, events)

    markers = widget._event_strip._markers
    marker_timestamps = sorted(m[2] for m in markers)
    assert marker_timestamps == [1_500_000_000, 1_800_000_000]  # exact recorded timestamps, unmodified


def test_event_details_are_preserved_verbatim(widget):
    events = [TimelineEvent(event_type="RADAR_DROPPED_FRAMES", description="dropped=15", timestamp_ns=1_500_000_000)]
    widget.set_session(1_000_000_000, 2_000_000_000, {"Camera": (1_000_000_000, 2_000_000_000)}, events)

    label = widget._event_strip._markers[0][3]
    assert label == "RADAR_DROPPED_FRAMES: dropped=15"
    # Not reinterpreted as "N frames dropped at this sample" -- the event
    # remains what the recorded metadata says: a single timestamped event.


def test_event_click_uses_seek_requested_signal_not_a_new_seek_mechanism(widget, qtbot):
    events = [TimelineEvent(event_type="RADAR_DROPPED_FRAMES", description="dropped=15", timestamp_ns=1_500_000_000)]
    widget.set_session(1_000_000_000, 2_000_000_000, {"Camera": (1_000_000_000, 2_000_000_000)}, events)
    widget.resize(400, 200)

    seen = []
    widget.seekRequested.connect(seen.append)
    widget._event_strip.markerClicked.emit(1_500_000_000)

    assert seen == [1_500_000_000]  # forwarded verbatim, same signal MainWindow wires to _seek_to_timestamp


def test_clicking_elsewhere_requests_details_not_a_seek(widget, qtbot):
    widget.resize(400, 200)
    seeks = []
    details = []
    widget.seekRequested.connect(seeks.append)
    widget.detailsRequested.connect(lambda: details.append(True))

    class FakeEvent:
        pass

    widget.mousePressEvent(FakeEvent())
    assert details == [True]
    assert seeks == []


# --------------------------------------------------------------------------- #
# Session switching / no stale state / no disk I/O.
# --------------------------------------------------------------------------- #

def test_set_session_replaces_previous_state_atomically(widget):
    widget.set_session(0, 1_000_000_000, {"Camera": (0, 1_000_000_000), "Radar": (0, 1_000_000_000)},
                        [TimelineEvent("RADAR_DROPPED_FRAMES", "dropped=1", 500_000_000)])
    widget.set_current(_snapshot(camera_ts=500_000_000))

    widget.set_session(2_000_000_000, 3_000_000_000, {"Camera": (2_000_000_000, 3_000_000_000)}, [])

    assert widget.coverage() == {"Camera": (2_000_000_000, 3_000_000_000)}
    assert "Radar" not in widget.coverage()
    assert widget._event_strip._markers == []
    assert widget._snapshot is None  # the old session's current-position state is gone, not carried over


def test_clear_resets_everything(widget):
    widget.set_session(0, 1_000_000_000, {"Camera": (0, 1_000_000_000)},
                        [TimelineEvent("RADAR_DROPPED_FRAMES", "x", 500_000_000)])
    widget.set_current(_snapshot())

    widget.clear()

    assert widget.coverage() == {}
    assert widget._event_strip._markers == []
    assert widget._snapshot is None


def test_set_session_and_set_current_perform_no_disk_io(widget, monkeypatch):
    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("TimelineDiagnostics must not perform disk I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        widget.set_session(0, 1_000_000_000, {"Camera": (0, 1_000_000_000), "Radar": (0, 1_000_000_000)},
                            [TimelineEvent("RADAR_DROPPED_FRAMES", "dropped=1", 500_000_000)])
        widget.set_current(_snapshot())
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


# --------------------------------------------------------------------------- #
# Boundary / degenerate camera ranges: no division-by-zero, no rendering
# exceptions, cursor always stays inside the axis.
# --------------------------------------------------------------------------- #

def test_cursor_at_first_camera_frame(widget):
    widget.set_session(1_000_000_000, 2_000_000_000, {"Camera": (1_000_000_000, 2_000_000_000)}, [])
    widget.set_current(_snapshot(camera_ts=1_000_000_000, frame_index=0))
    frac = widget._fraction(widget._diag_for_lane("Camera").sensor_timestamp_ns)
    assert frac == pytest.approx(0.0)
    widget.resize(400, 200)
    widget.repaint()  # must not raise


def test_cursor_at_last_camera_frame(widget):
    widget.set_session(1_000_000_000, 2_000_000_000, {"Camera": (1_000_000_000, 2_000_000_000)}, [])
    widget.set_current(_snapshot(camera_ts=2_000_000_000, frame_index=508))
    frac = widget._fraction(widget._diag_for_lane("Camera").sensor_timestamp_ns)
    assert frac == pytest.approx(1.0)
    widget.resize(400, 200)
    widget.repaint()  # must not raise


def test_degenerate_camera_range_single_timestamp_does_not_raise(widget):
    """camera_first_ns == camera_last_ns (e.g. a one-frame recording): the
    span must be guarded against zero without raising or dividing by zero."""
    widget.set_session(5_000_000_000, 5_000_000_000, {"Camera": (5_000_000_000, 5_000_000_000)}, [])
    widget.set_current(_snapshot(camera_ts=5_000_000_000, frame_index=0))
    widget.resize(400, 200)
    widget.repaint()  # must not raise, no ZeroDivisionError
    assert widget._fraction(5_000_000_000) is not None


def test_sensor_starting_before_camera_is_clipped_not_dropped(widget):
    """A sensor whose first recorded sample predates the camera's own first
    frame must still show a bar (clipped to the visible axis), not vanish
    and not be treated as an error."""
    widget.set_session(1_000_000_000, 2_000_000_000,
                        {"Camera": (1_000_000_000, 2_000_000_000), "Radar": (500_000_000, 1_500_000_000)}, [])
    assert widget.coverage()["Radar"] == (500_000_000, 1_500_000_000)  # exact recorded range, not clipped in the data
    widget.resize(400, 200)
    widget.repaint()  # the clipping happens only visually, never raises


def test_sensor_ending_after_camera_is_clipped_not_dropped(widget):
    widget.set_session(1_000_000_000, 2_000_000_000,
                        {"Camera": (1_000_000_000, 2_000_000_000), "Lidar": (1_500_000_000, 2_500_000_000)}, [])
    assert widget.coverage()["Lidar"] == (1_500_000_000, 2_500_000_000)
    widget.resize(400, 200)
    widget.repaint()


def test_one_sample_sensor_coverage_renders_without_raising(widget):
    """A sensor with exactly one recorded sample has first==last; the bar
    degenerates to a point/dash rather than raising or disappearing."""
    widget.set_session(0, 1_000_000_000, {"Camera": (0, 1_000_000_000), "MTi-8": (500_000_000, 500_000_000)}, [])
    widget.set_current(_snapshot(mti=SensorDiagnostic("MTi-8", SensorState.AVAILABLE, sample_index=0,
                                                        sensor_timestamp_ns=500_000_000, delta_ns=0, sync_class="GOOD")))
    widget.resize(400, 200)
    widget.repaint()
    assert widget.coverage()["MTi-8"] == (500_000_000, 500_000_000)


def test_very_short_recording_renders_without_raising(widget):
    widget.set_session(1_000_000_000, 1_000_050_000, {"Camera": (1_000_000_000, 1_000_050_000)}, [])
    widget.set_current(_snapshot(camera_ts=1_000_025_000))
    widget.resize(400, 200)
    widget.repaint()


# --------------------------------------------------------------------------- #
# Every SensorHealth state must be visually representable (a mapped color)
# without turning states into a score/ranking.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("state", [
    SensorState.AVAILABLE, SensorState.NO_DATA, SensorState.NO_FIX, SensorState.STALE,
    SensorState.DROPPED, SensorState.INVALID, SensorState.UNAVAILABLE,
])
def test_every_sensor_state_has_a_distinct_visual_treatment(widget, state):
    from timeline_diagnostics import _STATE_COLORS, _DEFAULT_STATE_COLOR

    color = _STATE_COLORS.get(state, _DEFAULT_STATE_COLOR)
    assert color is not None  # every state maps to *some* color, not a numeric score

    diag = SensorDiagnostic("Radar", state, sample_index=1, sensor_timestamp_ns=1_000_000_000, delta_ns=0,
                             sync_class="GOOD")
    widget.set_session(0, 2_000_000_000, {"Camera": (0, 2_000_000_000), "Radar": (0, 2_000_000_000)}, [])
    widget.set_current(_snapshot(radar=diag))
    widget.resize(400, 200)
    widget.repaint()  # every state must paint without raising


def test_state_colors_are_pairwise_distinguishable_not_a_ranking():
    from timeline_diagnostics import _STATE_COLORS

    # AVAILABLE/STALE/DROPPED are meant to look different from each other;
    # this is not an ordinal "better/worse" scale, just distinct colors.
    assert _STATE_COLORS[SensorState.AVAILABLE] != _STATE_COLORS[SensorState.STALE]
    assert _STATE_COLORS[SensorState.STALE] != _STATE_COLORS[SensorState.DROPPED]
    assert _STATE_COLORS[SensorState.AVAILABLE] != _STATE_COLORS[SensorState.DROPPED]


# --------------------------------------------------------------------------- #
# Resizing / narrow widths must never raise.
# --------------------------------------------------------------------------- #

def test_resize_wide_and_narrow_does_not_raise(widget):
    widget.set_session(0, 2_000_000_000, {"Camera": (0, 2_000_000_000), "Radar": (0, 2_000_000_000)}, [
        TimelineEvent("RADAR_DROPPED_FRAMES", "dropped=15", 500_000_000),
    ])
    widget.set_current(_snapshot(camera_ts=1_000_000_000))

    for w, h in [(2000, 300), (300, 300), (1, 1), (50, 50), (800, 120)]:
        widget.resize(w, h)
        widget.repaint()  # must never raise regardless of size


def test_narrow_widget_still_reports_correct_coverage(widget):
    widget.set_session(0, 2_000_000_000, {"Camera": (0, 2_000_000_000), "Radar": (0, 2_000_000_000)}, [])
    widget.resize(40, 200)
    widget.repaint()
    assert widget.coverage()["Radar"] == (0, 2_000_000_000)  # data is untouched by tiny rendering width


# --------------------------------------------------------------------------- #
# Hover / details perform no I/O and use only already-held data.
# --------------------------------------------------------------------------- #

def test_hover_over_sensor_lane_performs_no_disk_io(widget, monkeypatch):
    radar_diag = SensorDiagnostic("Radar", SensorState.STALE, sample_index=453,
                                   sensor_timestamp_ns=93_519_006, delta_ns=93_519_006, sync_class="STALE")
    widget.set_session(0, 1_000_000_000, {"Camera": (0, 1_000_000_000), "Radar": (0, 1_000_000_000)}, [])
    widget.set_current(_snapshot(radar=radar_diag))
    widget.resize(400, 200)
    _force_layout(widget)

    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("hover must not perform disk I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        lane = widget._lane_at_y(widget._lane_rects["Radar"][0] + 1)
        assert lane == "Radar"
        text = widget._tooltip_for(lane, widget._diag_for_lane(lane))
        assert "Radar" in text and "STALE" in text
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


def test_hover_tooltip_uses_only_existing_diagnostic_fields(widget):
    radar_diag = SensorDiagnostic("Radar", SensorState.AVAILABLE, sample_index=453,
                                   sensor_timestamp_ns=1_093_519_006, delta_ns=93_519_006, sync_class="WARNING")
    text = widget._tooltip_for("Radar", radar_diag)
    assert "453" in text
    assert "93.5" in text or "93.519" in text
    assert "AVAILABLE" in text
    assert "WARNING" in text


def test_event_hover_reads_existing_marker_data_no_new_event_path(widget, monkeypatch):
    events = [TimelineEvent("RADAR_DROPPED_FRAMES", "dropped=15", 500_000_000)]
    widget.set_session(0, 1_000_000_000, {"Camera": (0, 1_000_000_000)}, events)
    widget._event_strip.resize(400, 12)

    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("event hover must not perform disk I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        class FakeMoveEvent:
            def position(self):
                from PySide6.QtCore import QPointF
                return QPointF(200, 5)  # roughly the middle -> near the 0.5 fraction marker

            def globalPosition(self):
                from PySide6.QtCore import QPointF
                return QPointF(200, 5)

        widget._handle_event_strip_hover(FakeMoveEvent())  # must not raise or touch disk
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


# --------------------------------------------------------------------------- #
# MainWindow integration: coverage built from session arrays, stale results
# cannot overwrite the current timeline, existing EventMarkerStrip intact.
# --------------------------------------------------------------------------- #

class FakeMti:
    def __init__(self, timestamp_ns):
        self.timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64)

    @property
    def n(self):
        return len(self.timestamp_ns)


class FakeLidar:
    def __init__(self, timestamps_ns):
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)

    @property
    def num_scans(self):
        return len(self.timestamps_ns)


class FakeSession:
    def __init__(self, frames, radar=None, mti=None, lidar=None, events=None):
        self.path = Path("/nonexistent")
        self.session_id = "fake"
        self.frames = frames
        self.radar = radar if radar is not None else pd.DataFrame()
        self.mti = mti
        self.lidar = lidar
        self.events = events if events is not None else pd.DataFrame()


@pytest.fixture
def window(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)
    yield win
    win.close()


def test_mainwindow_setup_builds_coverage_from_already_loaded_arrays(window):
    frames = pd.DataFrame({"timestamp_ns": [1_000_000_000, 1_100_000_000, 1_200_000_000]})
    radar = pd.DataFrame({"timestamp_ns": [1_010_000_000, 1_190_000_000], "frame_id": [1, 2]})
    mti = FakeMti([990_000_000, 1_205_000_000])
    lidar = FakeLidar([995_000_000, 1_198_000_000])
    window.session = FakeSession(frames, radar=radar, mti=mti, lidar=lidar)
    window.svo = None

    window._setup_timeline_diagnostics()

    coverage = window.timeline_diagnostics.coverage()
    assert coverage["Camera"] == (1_000_000_000, 1_200_000_000)
    assert coverage["Radar"] == (1_010_000_000, 1_190_000_000)
    assert coverage["Lidar"] == (995_000_000, 1_198_000_000)
    assert coverage["MTi-8"] == (990_000_000, 1_205_000_000)
    assert coverage["GNSS"] == coverage["MTi-8"]  # GNSS mirrors MTi's own timestamps
    assert "ZED IMU" not in coverage  # no svo -> no fabricated lane


def test_mainwindow_wires_marker_seek_to_the_shared_seek_path(window, monkeypatch):
    # MainWindow._build_ui already connects timeline_diagnostics.seekRequested
    # straight to the existing _seek_to_timestamp -- verify that wiring, not a
    # second one made by this test.
    seen = []
    monkeypatch.setattr(window, "_seek_to_timestamp", lambda ts: seen.append(ts))

    window.timeline_diagnostics.seekRequested.emit(123456789)
    assert seen == [123456789]


def test_stale_session_load_result_cannot_replace_current_timeline(window, monkeypatch):
    from session_loader import SessionLoadResult

    frames_a = pd.DataFrame({"timestamp_ns": [0, 100]})
    window.session = FakeSession(frames_a)
    window._session_generation = 2  # a newer request has already been made

    monkeypatch.setattr(window, "_setup_timeline_diagnostics", lambda: None)
    window.timeline_diagnostics.set_session(0, 100, {"Camera": (0, 100)}, [])
    before = window.timeline_diagnostics.coverage()

    stale_frames = pd.DataFrame({"timestamp_ns": [5_000, 6_000]})
    stale_result = SessionLoadResult(
        generation=1, session_dir=Path("/nonexistent"), ok=True,
        session=FakeSession(stale_frames), svo=None, sync_index=None,
        validation_result=None, sensor_health=None,
    )
    window._on_session_load_finished(stale_result)  # generation 1 is obsolete

    assert window.timeline_diagnostics.coverage() == before  # untouched by the discarded result


def test_existing_event_marker_strip_widget_still_present_and_functional(window):
    from event_marker_strip import EventMarkerStrip

    assert isinstance(window.event_marker_strip, EventMarkerStrip)
    seen = []
    window.event_marker_strip.markerClicked.connect(seen.append)
    window.event_marker_strip.markerClicked.emit(42)
    assert seen == [42]


def test_update_sensor_health_feeds_timeline_without_disk_io(window, monkeypatch):
    frames = pd.DataFrame({"timestamp_ns": [0, 100, 200]})
    window.session = FakeSession(frames)
    window.sync_index = type("SI", (), {
        "radar_for": lambda self, i: -1, "lidar_for": lambda self, i: -1,
        "mti_for": lambda self, i: -1, "svo_for": lambda self, i: -1,
    })()
    from sensor_health import SensorHealthModel
    window.sensor_health = SensorHealthModel(window.session, window.sync_index)
    window.current_index = 1

    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("no disk I/O expected from _update_sensor_health")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        window._update_sensor_health()
    finally:
        monkeypatch.setattr(builtins, "open", real_open)

    assert window.timeline_diagnostics._snapshot is not None
    assert window.timeline_diagnostics._snapshot.camera_frame_index == 1


# --------------------------------------------------------------------------- #
# Real dataset regression: known recorded events and the known radar delta.
# --------------------------------------------------------------------------- #

_DATASET_ROOT = Path(__file__).parent.parent / "output_multi-record" / "output"
_REAL_SESSION_DIR = _DATASET_ROOT / "20260914_070614"


@pytest.mark.skipif(not _REAL_SESSION_DIR.exists(), reason="real recorded session not available")
def test_real_dataset_timeline_shows_known_recorded_events_and_radar_delta():
    from session import RecordingSession
    from sensor_health import SensorHealthModel
    from sync_index import SyncIndex

    session = RecordingSession.load(_REAL_SESSION_DIR)
    sync_index = SyncIndex.build(session, svo=None)
    health = SensorHealthModel(session, sync_index, svo=None)

    widget = TimelineDiagnostics()
    frames_ts = session.frames["timestamp_ns"].to_numpy()
    widget.set_session(int(frames_ts[0]), int(frames_ts[-1]), {"Camera": (int(frames_ts[0]), int(frames_ts[-1]))},
                        health.drop_events)

    labels = {m[3] for m in widget._event_strip._markers}
    assert "RADAR_DROPPED_FRAMES: dropped=15" in labels
    assert "LIDAR_DROPPED_FRAMES: dropped=41" in labels

    snap0 = health.snapshot(0)
    widget.set_current(snap0)
    # The known baseline from earlier verification of this recording: frame 0
    # radar delta is +93.5 ms. Must remain unchanged by this UI-only change.
    assert widget._diag_for_lane("Radar").delta_ms == pytest.approx(93.519006, abs=1e-3)
