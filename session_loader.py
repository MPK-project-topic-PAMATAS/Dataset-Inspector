"""Background worker that prepares a session off the Qt UI thread, so
opening or switching recordings does not freeze the application.

Performs exactly the steps MainWindow._load_session used to run
synchronously -- RecordingSession.load(), SvoReader construction (opening
the SVO2 and, if uncached, building its frame/timestamp index),
SyncIndex.build(), DatasetIntegrityValidator.validate(), and
SensorHealthModel construction -- in the same order, using the same
existing classes unmodified. It never changes what any of them compute; it
only moves *where* (which thread) they run.

Lives in its own dedicated QThread (see MainWindow), separate from
SensorWorker: SessionLoader's job is to *prepare* a session; SensorWorker's
job is to serve per-frame camera/radar requests for whichever session
MainWindow has since *committed* as active. Neither ever does the other's
job, and this module never touches a Qt widget -- only MainWindow, on the
UI thread, does that, once a SessionLoadResult arrives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from sensor_health import SensorHealthModel, SyncThresholds
from session import RecordingSession
from svo_reader import SvoReader
from sync_index import SyncIndex
from validator import DatasetIntegrityValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class SessionLoadRequest:
    generation: int
    session_dir: Path
    thresholds: SyncThresholds = field(default_factory=SyncThresholds)


@dataclass
class SessionLoadResult:
    """Everything needed to commit a newly-prepared session, or to report
    why preparing it failed. Holds references to the already-built objects
    (RecordingSession, SvoReader, SyncIndex, ...) -- never a copy of their
    underlying arrays.
    """

    generation: int
    session_dir: Path
    ok: bool
    session: Optional[RecordingSession] = None
    svo: Optional[SvoReader] = None
    sync_index: Optional[SyncIndex] = None
    validation_result: Optional[ValidationResult] = None
    sensor_health: Optional[SensorHealthModel] = None
    failed_stage: str = ""
    error: str = ""


class SessionLoader(QObject):
    """Lives in its own QThread. Processes one session-load request at a
    time -- exactly like SensorWorker processes one frame request at a
    time -- so requests naturally serialize without extra locking.
    """

    progress = Signal(int, str)  # generation, stage label (coarse; not a time estimate)
    finished = Signal(object)    # SessionLoadResult

    @Slot(object)
    def process(self, request: SessionLoadRequest) -> None:
        gen = request.generation
        session_dir = request.session_dir

        try:
            self.progress.emit(gen, "manifests / metadata")
            session = RecordingSession.load(session_dir)
        except Exception as e:
            logger.warning("session load failed (gen=%s) at RecordingSession.load: %s", gen, e)
            self.finished.emit(SessionLoadResult(
                gen, session_dir, ok=False, failed_stage="Session metadata/structure", error=str(e),
            ))
            return

        # A camera that fails to open is an existing, non-fatal condition
        # (the original _load_session caught this and left self.svo as
        # None, showing "Camera unavailable" -- preserved here verbatim).
        svo = None
        try:
            self.progress.emit(gen, "camera (SVO)")
            svo = SvoReader(session.svo_path)
        except Exception as e:
            logger.info("SVO unavailable for session %s (gen=%s): %s", session_dir, gen, e)

        try:
            self.progress.emit(gen, "synchronization index")
            sync_index = SyncIndex.build(session, svo)
        except Exception as e:
            logger.warning("session load failed (gen=%s) at SyncIndex.build: %s", gen, e)
            if svo is not None:
                svo.close()
            self.finished.emit(SessionLoadResult(
                gen, session_dir, ok=False, failed_stage="Synchronization index", error=str(e),
            ))
            return

        # Validation and sensor-health init are, like in the original
        # synchronous code, non-fatal: a failure here still leaves a usable
        # session, just without that one diagnostic.
        validation_result = None
        try:
            self.progress.emit(gen, "dataset integrity validation")
            validation_result = DatasetIntegrityValidator(session, svo).validate()
        except Exception as e:
            logger.warning("dataset integrity validation failed (gen=%s): %s", gen, e)

        sensor_health = None
        try:
            self.progress.emit(gen, "sensor health")
            sensor_health = SensorHealthModel(session, sync_index, svo, validation_result, request.thresholds)
        except Exception as e:
            logger.warning("sensor health init failed (gen=%s): %s", gen, e)

        self.progress.emit(gen, "ready")
        self.finished.emit(SessionLoadResult(
            gen, session_dir, ok=True, session=session, svo=svo, sync_index=sync_index,
            validation_result=validation_result, sensor_health=sensor_health,
        ))
