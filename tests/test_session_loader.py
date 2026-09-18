"""Tests for the background SessionLoader (session_loader.py) and its
generation-based supersession wiring in MainWindow.

These cover the loader in isolation (successful load, captured exceptions)
and the full race scenarios explicitly required: a superseded in-flight
load must never become active, a failed load must never replace a good
active session, and progress/finished signals for an obsolete generation
must be ignored. A real-dataset check lives at the bottom (skipped if the
recording isn't present).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from session_loader import SessionLoader, SessionLoadRequest, SessionLoadResult


class FakeSvo:
    def __init__(self):
        self.closed = False
        self.timestamps_ns = None
        self.has_zed_imu = False

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, session_id="fake"):
        self.session_id = session_id
        self.path = Path("/nonexistent")
        self.frames = pd.DataFrame({"timestamp_ns": [0]})
        self.radar = pd.DataFrame()
        self.events = pd.DataFrame()
        self.mti = None
        self.lidar = None
        self.manifest = {}
        self.session_manifest = {}

    @property
    def svo_path(self):
        return self.path / "raw" / "recording.svo2"


def _result(generation, ok=True, session=None, svo=None, sync_index=None,
            validation_result=None, sensor_health=None, failed_stage="", error=""):
    return SessionLoadResult(
        generation=generation, session_dir=Path("/nonexistent"), ok=ok,
        session=session, svo=svo, sync_index=sync_index,
        validation_result=validation_result, sensor_health=sensor_health,
        failed_stage=failed_stage, error=error,
    )


# --------------------------------------------------------------------------- #
# SessionLoader in isolation.
# --------------------------------------------------------------------------- #

def test_session_loads_successfully_in_worker(tmp_path, monkeypatch):
    import session_loader as sl_module

    fake_session = FakeSession()
    fake_svo = FakeSvo()
    monkeypatch.setattr(sl_module, "RecordingSession", type("RS", (), {"load": staticmethod(lambda p: fake_session)}))
    monkeypatch.setattr(sl_module, "SvoReader", lambda path: fake_svo)
    monkeypatch.setattr(sl_module, "SyncIndex", type("SI", (), {"build": staticmethod(lambda s, v: "sync-index")}))
    monkeypatch.setattr(sl_module, "DatasetIntegrityValidator", lambda s, v: type("V", (), {"validate": lambda self: "validation-result"})())
    monkeypatch.setattr(sl_module, "SensorHealthModel", lambda *a, **k: "sensor-health")

    loader = SessionLoader()
    results = []
    loader.finished.connect(results.append)

    loader.process(SessionLoadRequest(generation=1, session_dir=tmp_path))

    assert len(results) == 1
    result = results[0]
    assert result.ok is True
    assert result.generation == 1
    assert result.session is fake_session
    assert result.svo is fake_svo
    assert result.sync_index == "sync-index"
    assert result.validation_result == "validation-result"
    assert result.sensor_health == "sensor-health"


def test_loader_captures_recording_session_load_exception():
    import session_loader as sl_module

    class RaisingRS:
        @staticmethod
        def load(path):
            raise RuntimeError("simulated: cannot read manifest")

    orig = sl_module.RecordingSession
    sl_module.RecordingSession = RaisingRS
    try:
        loader = SessionLoader()
        results = []
        loader.finished.connect(results.append)
        loader.process(SessionLoadRequest(generation=1, session_dir=Path("/nonexistent")))
    finally:
        sl_module.RecordingSession = orig

    assert len(results) == 1
    result = results[0]
    assert result.ok is False
    assert "manifest" in result.error
    assert result.failed_stage == "Session metadata/structure"
    assert result.session is None


def test_loader_captures_sync_index_exception_and_closes_svo(monkeypatch):
    import session_loader as sl_module

    fake_session = FakeSession()
    fake_svo = FakeSvo()
    monkeypatch.setattr(sl_module, "RecordingSession", type("RS", (), {"load": staticmethod(lambda p: fake_session)}))
    monkeypatch.setattr(sl_module, "SvoReader", lambda path: fake_svo)

    class RaisingSyncIndex:
        @staticmethod
        def build(session, svo):
            raise RuntimeError("simulated: bad timestamps")

    monkeypatch.setattr(sl_module, "SyncIndex", RaisingSyncIndex)

    loader = SessionLoader()
    results = []
    loader.finished.connect(results.append)
    loader.process(SessionLoadRequest(generation=5, session_dir=Path("/nonexistent")))

    assert len(results) == 1
    result = results[0]
    assert result.ok is False
    assert result.failed_stage == "Synchronization index"
    assert fake_svo.closed is True  # never left dangling on a failed load


def test_svo_open_failure_is_non_fatal(monkeypatch):
    """A camera that fails to open must not fail the whole session load --
    matches the original synchronous behavior (self.svo stayed None)."""
    import session_loader as sl_module

    fake_session = FakeSession()
    monkeypatch.setattr(sl_module, "RecordingSession", type("RS", (), {"load": staticmethod(lambda p: fake_session)}))

    def raising_svo(path):
        raise RuntimeError("no camera")

    monkeypatch.setattr(sl_module, "SvoReader", raising_svo)
    monkeypatch.setattr(sl_module, "SyncIndex", type("SI", (), {"build": staticmethod(lambda s, v: "sync-index")}))
    monkeypatch.setattr(sl_module, "DatasetIntegrityValidator", lambda s, v: type("V", (), {"validate": lambda self: None})())
    monkeypatch.setattr(sl_module, "SensorHealthModel", lambda *a, **k: None)

    loader = SessionLoader()
    results = []
    loader.finished.connect(results.append)
    loader.process(SessionLoadRequest(generation=1, session_dir=Path("/nonexistent")))

    assert results[0].ok is True
    assert results[0].svo is None


def test_validation_and_sensor_health_failures_are_non_fatal(monkeypatch):
    import session_loader as sl_module

    fake_session = FakeSession()
    fake_svo = FakeSvo()
    monkeypatch.setattr(sl_module, "RecordingSession", type("RS", (), {"load": staticmethod(lambda p: fake_session)}))
    monkeypatch.setattr(sl_module, "SvoReader", lambda path: fake_svo)
    monkeypatch.setattr(sl_module, "SyncIndex", type("SI", (), {"build": staticmethod(lambda s, v: "sync-index")}))

    def raising_validator(s, v):
        raise RuntimeError("validator boom")

    def raising_health(*a, **k):
        raise RuntimeError("health boom")

    monkeypatch.setattr(sl_module, "DatasetIntegrityValidator", raising_validator)
    monkeypatch.setattr(sl_module, "SensorHealthModel", raising_health)

    loader = SessionLoader()
    results = []
    loader.finished.connect(results.append)
    loader.process(SessionLoadRequest(generation=1, session_dir=Path("/nonexistent")))

    result = results[0]
    assert result.ok is True  # session is still usable
    assert result.validation_result is None
    assert result.sensor_health is None


def test_progress_emitted_with_stage_labels_not_percentages(monkeypatch):
    import session_loader as sl_module

    fake_session = FakeSession()
    fake_svo = FakeSvo()
    monkeypatch.setattr(sl_module, "RecordingSession", type("RS", (), {"load": staticmethod(lambda p: fake_session)}))
    monkeypatch.setattr(sl_module, "SvoReader", lambda path: fake_svo)
    monkeypatch.setattr(sl_module, "SyncIndex", type("SI", (), {"build": staticmethod(lambda s, v: "sync-index")}))
    monkeypatch.setattr(sl_module, "DatasetIntegrityValidator", lambda s, v: type("V", (), {"validate": lambda self: None})())
    monkeypatch.setattr(sl_module, "SensorHealthModel", lambda *a, **k: None)

    loader = SessionLoader()
    stages = []
    loader.progress.connect(lambda gen, stage: stages.append((gen, stage)))
    loader.process(SessionLoadRequest(generation=3, session_dir=Path("/nonexistent")))

    assert all(gen == 3 for gen, _ in stages)
    assert all(isinstance(stage, str) and not stage.endswith("%") for _, stage in stages)
    assert "ready" in [stage for _, stage in stages]


# --------------------------------------------------------------------------- #
# MainWindow: generation-based supersession and commit/discard semantics.
# --------------------------------------------------------------------------- #

@pytest.fixture
def window(qtbot):
    from main_window import MainWindow

    win = MainWindow(dataset_root=None)
    qtbot.addWidget(win)
    yield win
    win.close()


def test_ui_receives_successful_result_and_commits(window, monkeypatch):
    committed = []
    monkeypatch.setattr(window, "_commit_session", lambda result: committed.append(result))

    window._session_generation = 1
    result = _result(1, ok=True, session=FakeSession("A"))
    window._on_session_load_finished(result)

    assert committed == [result]


def test_failed_session_does_not_replace_active_session(window, monkeypatch):
    active = FakeSession("A")
    window.session = active
    window._session_generation = 2

    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)
    window._on_session_load_finished(_result(2, ok=False, failed_stage="Lidar initialization", error="boom"))

    assert window.session is active  # unchanged


def test_failed_first_load_leaves_no_session_without_crashing(window, monkeypatch):
    window.session = None
    window._session_generation = 1
    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)

    window._on_session_load_finished(_result(1, ok=False, failed_stage="Session metadata/structure", error="boom"))

    assert window.session is None  # no crash, simply nothing loaded yet


def test_newer_generation_supersedes_older_result(window):
    window.session = FakeSession("A")
    window._session_generation = 2  # a second request (B->C is generation 2) already made

    stale_svo = FakeSvo()
    committed = []
    window._commit_session = lambda result: committed.append(result)

    window._on_session_load_finished(_result(1, ok=True, session=FakeSession("B"), svo=stale_svo))

    assert committed == []  # never committed
    assert stale_svo.closed is True  # its SvoReader was safely closed instead


def test_generation_mismatch_on_failed_result_is_also_discarded_silently(window):
    window.session = FakeSession("A")
    window._session_generation = 2

    # A failure for an already-superseded generation must not even pop a dialog
    # complaining about "B" -- it's simply irrelevant now.
    window._on_session_load_finished(_result(1, ok=False, failed_stage="X", error="boom"))

    assert window.session.session_id == "A"


def test_progress_signal_for_obsolete_generation_is_ignored(window):
    window.session = FakeSession("A")
    window._session_generation = 2
    window._loading_session_name = "C"
    window.info_label.setText("untouched")

    window._on_session_load_progress(1, "lidar initialization")  # generation 1 is stale

    assert window.info_label.text() == "untouched"


def test_progress_signal_for_current_generation_updates_label(window):
    window.session = None
    window._session_generation = 5
    window._loading_session_name = "C"

    window._on_session_load_progress(5, "lidar initialization")

    assert "lidar initialization" in window.info_label.text()
    assert "C" in window.info_label.text()


# --------------------------------------------------------------------------- #
# The exact races from the design brief.
# --------------------------------------------------------------------------- #

def _spy_on_commit(window, monkeypatch):
    """Lets a real _commit_session run (so all its invariants still apply)
    while also recording, in order, which session_id ended up committed."""
    committed_ids = []
    real_commit = window._commit_session

    def spy(result):
        real_commit(result)
        committed_ids.append(result.session.session_id)

    monkeypatch.setattr(window, "_commit_session", spy)
    # The bulk of _commit_session's tail is widget wiring irrelevant to
    # these generation/race tests -- stub it so a FakeSession can pass
    # through without needing real frames/radar/etc.
    for name in ("_stop_playback", "_await_worker_idle", "_plot_imu_series", "_plot_zed_imu_series",
                 "_plot_gnss_track", "_populate_events", "_show_lidar_status",
                 "_display_validation_result", "_populate_event_markers",
                 "_setup_timeline_diagnostics", "_seek"):
        monkeypatch.setattr(window, name, lambda *a, **k: None)
    return committed_ids


def test_race_b_superseded_by_c_before_either_finishes_then_b_then_c(window, monkeypatch):
    """A active; request B; request C before B finishes; B finishes; C
    finishes. Expected: A -> C directly. B must never become active."""
    window.session = FakeSession("A")
    committed_ids = _spy_on_commit(window, monkeypatch)

    window._session_generation = 1  # "request B"
    window._session_generation = 2  # "request C" arrives before B finishes; B is now obsolete

    b_svo = FakeSvo()
    window._on_session_load_finished(_result(1, ok=True, session=FakeSession("B"), svo=b_svo))
    assert window.session.session_id == "A"  # B never became active
    assert b_svo.closed is True

    c_session = FakeSession("C")
    window._on_session_load_finished(_result(2, ok=True, session=c_session, svo=FakeSvo()))
    assert window.session is c_session  # C is now active

    # The visible sequence of *committed* sessions was exactly A -> C.
    assert committed_ids == ["C"]


def test_race_c_finishes_before_b_b_then_discarded(window, monkeypatch):
    """Same requests, opposite completion order: C finishes first (becomes
    active), then B's late result arrives and must still be discarded."""
    window.session = FakeSession("A")
    committed_ids = _spy_on_commit(window, monkeypatch)

    window._session_generation = 1  # "request B"
    window._session_generation = 2  # "request C"

    c_session = FakeSession("C")
    window._on_session_load_finished(_result(2, ok=True, session=c_session, svo=FakeSvo()))
    assert window.session is c_session

    b_svo = FakeSvo()
    window._on_session_load_finished(_result(1, ok=True, session=FakeSession("B"), svo=b_svo))
    assert window.session is c_session  # still C -- B did not overwrite it
    assert b_svo.closed is True

    assert committed_ids == ["C"]


def test_failure_race_b_fails_c_succeeds(window, monkeypatch):
    """A active; request B; request C; B fails; C succeeds. Expected: A -> C.
    B must never become active."""
    window.session = FakeSession("A")
    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)
    committed_ids = _spy_on_commit(window, monkeypatch)

    window._session_generation = 1  # "request B"
    window._session_generation = 2  # "request C"

    window._on_session_load_finished(_result(1, ok=False, failed_stage="Lidar initialization", error="bad pcap"))
    assert window.session.session_id == "A"

    c_session = FakeSession("C")
    window._on_session_load_finished(_result(2, ok=True, session=c_session, svo=FakeSvo()))
    assert window.session is c_session
    assert committed_ids == ["C"]


def test_failure_race_single_failed_request_leaves_active_session_untouched(window, monkeypatch):
    """A active; request B; B fails. Expected: A remains active."""
    window.session = FakeSession("A")
    monkeypatch.setattr("main_window.QMessageBox.warning", lambda *a, **k: None)

    window._session_generation = 1  # "request B"
    window._on_session_load_finished(_result(1, ok=False, failed_stage="Session metadata/structure", error="bad manifest"))
    assert window.session.session_id == "A"


# --------------------------------------------------------------------------- #
# Commit-time invariants: cache reset, SvoReader ownership, and that
# SyncIndex/validator/sensor-health belong to exactly the committed session.
# --------------------------------------------------------------------------- #

def test_commit_resets_cache_and_closes_old_svo_after_await_idle(window, monkeypatch):
    old_svo = FakeSvo()
    window.svo = old_svo
    window.session = FakeSession("A")

    calls = []
    monkeypatch.setattr(window, "_await_worker_idle", lambda: calls.append("await_idle"))
    window.resetCache.connect(lambda: calls.append("reset_cache"))
    # Stub out the widget-heavy tail of _commit_session -- irrelevant here.
    monkeypatch.setattr(window, "_plot_imu_series", lambda: None)
    monkeypatch.setattr(window, "_plot_zed_imu_series", lambda: None)
    monkeypatch.setattr(window, "_plot_gnss_track", lambda: None)
    monkeypatch.setattr(window, "_populate_events", lambda: None)
    monkeypatch.setattr(window, "_show_lidar_status", lambda: None)
    monkeypatch.setattr(window, "_display_validation_result", lambda: None)
    monkeypatch.setattr(window, "_populate_event_markers", lambda: None)
    monkeypatch.setattr(window, "_setup_timeline_diagnostics", lambda: None)
    monkeypatch.setattr(window, "_seek", lambda i: None)

    new_svo = FakeSvo()
    new_session = FakeSession("B")
    result = _result(1, ok=True, session=new_session, svo=new_svo,
                      sync_index="B-sync", validation_result="B-validation", sensor_health="B-health")

    window._commit_session(result)

    # await_idle happens before the cache reset and before the swap, exactly
    # preserving the ordering _load_session always used.
    assert calls.index("await_idle") < calls.index("reset_cache")
    assert old_svo.closed is True
    assert window.svo is new_svo
    assert window.session is new_session
    assert window.sync_index == "B-sync"
    assert window.validation_result == "B-validation"
    assert window.sensor_health == "B-health"


def test_commit_does_not_close_svo_if_unchanged(window, monkeypatch):
    """If the new result reuses the same SvoReader instance (not expected in
    practice, but the guard exists), it must not be closed out from under
    itself."""
    shared_svo = FakeSvo()
    window.svo = shared_svo
    window.session = FakeSession("A")

    monkeypatch.setattr(window, "_await_worker_idle", lambda: None)
    for name in ("_plot_imu_series", "_plot_zed_imu_series", "_plot_gnss_track",
                 "_populate_events", "_show_lidar_status", "_display_validation_result",
                 "_populate_event_markers", "_setup_timeline_diagnostics"):
        monkeypatch.setattr(window, name, lambda: None)
    monkeypatch.setattr(window, "_seek", lambda i: None)

    result = _result(1, ok=True, session=FakeSession("B"), svo=shared_svo)
    window._commit_session(result)

    assert shared_svo.closed is False


# --------------------------------------------------------------------------- #
# No widget manipulation from the loader thread (static check).
# --------------------------------------------------------------------------- #

def test_loader_module_never_touches_qt_widgets():
    import inspect
    import session_loader as sl_module

    source = inspect.getsource(sl_module)
    forbidden = ["setPixmap(", "setImage(", "setData(", "QMessageBox", "QLabel(", "QWidget"]
    hits = [token for token in forbidden if token in source]
    assert hits == [], f"session_loader.py must never manipulate widgets, found: {hits}"


# --------------------------------------------------------------------------- #
# Real dataset.
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
def test_real_session_loads_successfully_through_loader():
    session_dir = _first_real_session_dir()
    loader = SessionLoader()
    results = []
    loader.finished.connect(results.append)

    loader.process(SessionLoadRequest(generation=1, session_dir=session_dir))

    assert len(results) == 1
    result = results[0]
    assert result.ok is True
    assert result.session is not None
    assert result.sync_index is not None
    if result.svo is not None:
        result.svo.close()
