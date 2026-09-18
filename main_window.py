from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEventLoop, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from event_marker_strip import EventMarkerStrip
from lidar_3d_view import Lidar3DView
from map_view import MapView
from radar_3d_view import RadarPointCloud3DView
from radar_causal import causal_indices_for_frames
from sensor_health import SensorHealthModel, SensorHealthSnapshot, SyncThresholds
from session import RecordingSession
from session_loader import SessionLoadRequest, SessionLoadResult, SessionLoader
from svo_reader import SvoReader
from sync_index import MISSING, SyncIndex
from timeline_diagnostics import TimelineDiagnostics
from validator import DatasetIntegrityValidator, ValidationResult
from worker import FrameRequest, SensorWorker

RADAR_VIEW_MODES = ("2D", "3D")
RADAR_TEMPORAL_MODES = ("CAUSAL", "NEAREST")

logger = logging.getLogger(__name__)

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
WORKER_IDLE_TIMEOUT_MS = 3000


def ns_to_str(ts_ns: int) -> str:
    s = ts_ns / 1e9
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def _scrollable(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    return scroll


class MainWindow(QMainWindow):
    # Cross-thread dispatch: emitting this queues process() onto the worker
    # thread's event loop (Qt does this automatically for cross-thread
    # signal/slot connections), so no direct call ever touches the worker's
    # SvoReader/RecordingSession from the UI thread.
    requestFrame = Signal(object)
    # Emitted on session switch to drop the worker's camera/radar caches
    # before any frame from the new session is requested (see _load_session).
    resetCache = Signal()
    # Cross-thread dispatch to the SessionLoader (a separate worker/thread
    # from SensorWorker -- see session_loader.py): queues process() onto
    # the loader thread's event loop.
    requestSessionLoad = Signal(object)

    def __init__(self, dataset_root: Path | None = None):
        super().__init__()
        self.setWindowTitle("Dataset Inspector")
        self.resize(1500, 950)

        self.dataset_root = dataset_root
        self.session: RecordingSession | None = None
        self.svo: SvoReader | None = None
        self.sync_index: SyncIndex | None = None
        self.validation_result: ValidationResult | None = None
        self.sensor_health: SensorHealthModel | None = None
        self._current_health_snapshot: SensorHealthSnapshot | None = None
        # Display-only thresholds for the sensor-health strip/detail panel
        # (see sensor_health.SyncThresholds) -- never touched by SyncIndex,
        # never affects which sample gets selected or what data is shown.
        self._sync_thresholds = SyncThresholds()
        self.current_index = 0
        self.playing = False
        self.speed = 1.0
        self._heatmap_shape = None

        # Radar display-only state (see radar_causal.py's module docstring
        # for the exact NEAREST-vs-CAUSAL distinction). Neither of these
        # ever changes which recorded sample SyncIndex would select, nor
        # touches SensorHealthModel -- they only control what this window
        # currently *shows*. `_radar_causal_index` mirrors
        # SyncIndex.radar_index's shape/semantics (camera frame -> positional
        # row in session.radar, or MISSING) but is computed by the separate
        # causal rule and stored here, not inside SyncIndex.
        self._radar_view_mode = RADAR_VIEW_MODES[0]  # "2D" default -- preserves the existing view on first launch
        self._radar_temporal_mode = RADAR_TEMPORAL_MODES[0]  # "CAUSAL" default (see design brief §7/13)
        self._radar_ts = np.array([], dtype=np.int64)
        self._radar_causal_index = np.array([], dtype=np.int64)
        self._last_radar_selection = (None, None, False, False)  # (row, delta_ns, is_future, no_causal_yet)

        # Async camera/radar I/O bookkeeping. `_generation` increments on
        # every seek; a worker result is only applied if its generation
        # still matches, so a late result for a frame the user has since
        # scrubbed away from can never overwrite the current view. Only one
        # request is ever in flight and at most one more is coalesced
        # (`_pending_request`), so scrubbing can't pile up unbounded work.
        self._generation = 0
        self._worker_busy = False
        self._pending_request: FrameRequest | None = None

        # Session-loading bookkeeping, entirely separate from the above
        # per-frame `_generation` counter. `_session_generation` increments
        # on every session-load *request* (open dataset root, pick a
        # different session in the combo); a SessionLoadResult is only
        # committed if its generation still matches, so a session B whose
        # background load finishes after the user has already requested C
        # can never become active (see _on_session_load_finished).
        self._session_generation = 0
        self._loading_session_name = ""
        self._session_load_state = "IDLE"  # IDLE | LOADING | READY | FAILED

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)

        self._worker_thread = QThread(self)
        self._worker = SensorWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.start()

        self.requestFrame.connect(self._worker.process)
        self.resetCache.connect(self._worker.reset_caches)
        self._worker.cameraReady.connect(self._on_camera_ready)
        self._worker.radarReady.connect(self._on_radar_ready)
        self._worker.errorOccurred.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)

        # A second, separate dedicated thread/worker (see session_loader.py)
        # that only ever prepares sessions -- never serves camera/radar
        # frame requests, and is never touched by SensorWorker.
        self._session_loader_thread = QThread(self)
        self._session_loader = SessionLoader()
        self._session_loader.moveToThread(self._session_loader_thread)
        self._session_loader_thread.start()

        self.requestSessionLoad.connect(self._session_loader.process)
        self._session_loader.progress.connect(self._on_session_load_progress)
        self._session_loader.finished.connect(self._on_session_load_finished)

        self._build_ui()
        if dataset_root is not None:
            self._populate_sessions(dataset_root)

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        top = QHBoxLayout()
        self.open_btn = QPushButton("Open dataset root…")
        self.open_btn.clicked.connect(self._choose_root)
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self._on_session_selected)
        self.info_label = QLabel("No session loaded")
        top.addWidget(self.open_btn)
        top.addWidget(self.session_combo)
        top.addWidget(self.info_label, stretch=1)
        outer.addLayout(top)

        # --- dataset integrity panel: a concise status row + a details dialog ---
        integrity_row = QHBoxLayout()
        integrity_row.addWidget(QLabel("DATASET INTEGRITY:"))
        self.integrity_label = QLabel("not yet validated")
        integrity_row.addWidget(self.integrity_label, stretch=1)
        self.integrity_details_btn = QPushButton("Details…")
        self.integrity_details_btn.clicked.connect(self._show_integrity_details)
        integrity_row.addWidget(self.integrity_details_btn)
        outer.addLayout(integrity_row)

        # --- sensor health: position-aware ("what's available right now?"),
        # distinct from Dataset Integrity ("is the recording structurally
        # usable?") -- see sensor_health.py.
        health_row = QHBoxLayout()
        health_row.addWidget(QLabel("SENSOR STATUS:"))
        self.sensor_status_label = QLabel("not yet available")
        health_row.addWidget(self.sensor_status_label, stretch=1)
        self.sensor_status_details_btn = QPushButton("Details…")
        self.sensor_status_details_btn.clicked.connect(self._show_sensor_health_details)
        health_row.addWidget(self.sensor_status_details_btn)
        outer.addLayout(health_row)

        # --- timeline diagnostics: sensor coverage, current sync position,
        # and recorded drop events relative to the camera master timeline.
        # Purely a visualization over data MainWindow already has (session,
        # SyncIndex-selected samples via SensorHealthSnapshot, recorded
        # events) -- see timeline_diagnostics.py.
        outer.addWidget(QLabel("TIMELINE"))
        self.timeline_diagnostics = TimelineDiagnostics()
        self.timeline_diagnostics.seekRequested.connect(self._seek_to_timestamp)
        self.timeline_diagnostics.detailsRequested.connect(self._show_timeline_details)
        outer.addWidget(self.timeline_diagnostics)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # --- camera panel (both left and right, stacked) ---
        cam_panel = QWidget()
        cam_layout = QVBoxLayout(cam_panel)
        cam_layout.addWidget(QLabel("Left"))
        self.left_label = QLabel("Camera: no data")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setMinimumSize(420, 236)
        self.left_label.setStyleSheet("background: #111; color: #888;")
        cam_layout.addWidget(self.left_label, stretch=1)
        cam_layout.addWidget(QLabel("Right"))
        self.right_label = QLabel("Camera: no data")
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setMinimumSize(420, 236)
        self.right_label.setStyleSheet("background: #111; color: #888;")
        cam_layout.addWidget(self.right_label, stretch=1)
        splitter.addWidget(cam_panel)

        # --- radar + lidar panel ---
        radar_panel = QWidget()
        radar_layout = QVBoxLayout(radar_panel)
        radar_layout.addWidget(QLabel("Radar range-Doppler heatmap"))
        self.heatmap_view = pg.ImageView()
        self.heatmap_view.ui.roiBtn.hide()
        self.heatmap_view.ui.menuBtn.hide()
        self.heatmap_view.setMinimumHeight(220)
        radar_layout.addWidget(self.heatmap_view, stretch=1)
        # --- radar view/temporal-mode controls + header -------------------
        # Both the 2D and 3D widgets below are created exactly once (never
        # recreated on toggle -- see _on_radar_view_mode_changed) and both
        # are always kept in sync with the same selected radar sample (see
        # _on_radar_ready) -- see this change's design brief §5/§6.
        radar_controls = QHBoxLayout()
        radar_controls.addWidget(QLabel("Radar view:"))
        self.radar_view_2d_btn = QPushButton("2D")
        self.radar_view_3d_btn = QPushButton("3D")
        for btn in (self.radar_view_2d_btn, self.radar_view_3d_btn):
            btn.setCheckable(True)
        self.radar_view_2d_btn.setChecked(self._radar_view_mode == "2D")
        self.radar_view_3d_btn.setChecked(self._radar_view_mode == "3D")
        self._radar_view_group = QButtonGroup(self)
        self._radar_view_group.setExclusive(True)
        self._radar_view_group.addButton(self.radar_view_2d_btn)
        self._radar_view_group.addButton(self.radar_view_3d_btn)
        self.radar_view_2d_btn.clicked.connect(lambda: self._on_radar_view_mode_changed("2D"))
        self.radar_view_3d_btn.clicked.connect(lambda: self._on_radar_view_mode_changed("3D"))
        radar_controls.addWidget(self.radar_view_2d_btn)
        radar_controls.addWidget(self.radar_view_3d_btn)

        radar_controls.addSpacing(16)
        radar_controls.addWidget(QLabel("Radar time:"))
        self.radar_causal_btn = QPushButton("Causal")
        self.radar_nearest_btn = QPushButton("Nearest")
        for btn in (self.radar_causal_btn, self.radar_nearest_btn):
            btn.setCheckable(True)
        self.radar_causal_btn.setChecked(self._radar_temporal_mode == "CAUSAL")
        self.radar_nearest_btn.setChecked(self._radar_temporal_mode == "NEAREST")
        self._radar_temporal_group = QButtonGroup(self)
        self._radar_temporal_group.setExclusive(True)
        self._radar_temporal_group.addButton(self.radar_causal_btn)
        self._radar_temporal_group.addButton(self.radar_nearest_btn)
        self.radar_causal_btn.clicked.connect(lambda: self._on_radar_temporal_mode_changed("CAUSAL"))
        self.radar_nearest_btn.clicked.connect(lambda: self._on_radar_temporal_mode_changed("NEAREST"))
        radar_controls.addWidget(self.radar_causal_btn)
        radar_controls.addWidget(self.radar_nearest_btn)
        radar_controls.addStretch(1)
        radar_layout.addLayout(radar_controls)

        self.radar_header_label = QLabel("RADAR\n(no session loaded)")
        self.radar_header_label.setStyleSheet("color: #ccc;")
        self.radar_header_label.setToolTip(
            "The SensorHealth state shown here always reflects the existing NEAREST-mode "
            "radar diagnostic (SensorHealthModel is never mutated); it may differ from the "
            "sample currently displayed when Radar time = CAUSAL."
        )
        radar_layout.addWidget(self.radar_header_label)

        radar_layout.addWidget(QLabel("Radar detections (top-down, x/y meters, or 3D -- see toggle above)"))
        self.points_plot = pg.PlotWidget()
        self.points_plot.setAspectLocked(True)
        self.points_plot.setLabel("bottom", "x (m)")
        self.points_plot.setLabel("left", "y (m, forward)")
        self.points_plot.setMinimumHeight(220)
        self.points_scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 80, 80, 200))
        self.points_plot.addItem(self.points_scatter)

        self.radar_3d_view = RadarPointCloud3DView()
        self.radar_3d_view.setMinimumHeight(220)

        # QStackedWidget: both widgets exist for the whole session, toggling
        # only changes which one is current -- no OpenGL context churn, no
        # re-fetch (see _on_radar_view_mode_changed).
        self.radar_view_stack = QStackedWidget()
        self.radar_view_stack.addWidget(self.points_plot)   # index 0 = 2D
        self.radar_view_stack.addWidget(self.radar_3d_view)  # index 1 = 3D
        self.radar_view_stack.setCurrentIndex(RADAR_VIEW_MODES.index(self._radar_view_mode))
        radar_layout.addWidget(self.radar_view_stack, stretch=1)

        radar_layout.addWidget(QLabel("Lidar (Ouster) interactive 3D point cloud"))
        self.lidar_view = Lidar3DView()
        radar_layout.addWidget(self.lidar_view, stretch=1)
        splitter.addWidget(_scrollable(radar_panel))

        # --- imu / gnss panel ---
        imu_panel = QWidget()
        imu_layout = QVBoxLayout(imu_panel)
        imu_layout.addWidget(QLabel("MTi-8 IMU acceleration (m/s^2)"))
        self.accel_plot = pg.PlotWidget()
        self.accel_plot.setMinimumHeight(160)
        self.accel_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y"))
        self.accel_plot.addItem(self.accel_cursor)
        imu_layout.addWidget(self.accel_plot)

        imu_layout.addWidget(QLabel("MTi-8 IMU angular velocity (rad/s)"))
        self.gyro_plot = pg.PlotWidget()
        self.gyro_plot.setMinimumHeight(160)
        self.gyro_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y"))
        self.gyro_plot.addItem(self.gyro_cursor)
        imu_layout.addWidget(self.gyro_plot)

        imu_layout.addWidget(QLabel("ZED 2 onboard IMU acceleration (m/s^2)"))
        self.zed_accel_plot = pg.PlotWidget()
        self.zed_accel_plot.setMinimumHeight(160)
        self.zed_accel_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y"))
        self.zed_accel_plot.addItem(self.zed_accel_cursor)
        imu_layout.addWidget(self.zed_accel_plot)

        imu_layout.addWidget(QLabel("ZED 2 onboard IMU angular velocity (rad/s)"))
        self.zed_gyro_plot = pg.PlotWidget()
        self.zed_gyro_plot.setMinimumHeight(160)
        self.zed_gyro_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y"))
        self.zed_gyro_plot.addItem(self.zed_gyro_cursor)
        imu_layout.addWidget(self.zed_gyro_plot)

        imu_layout.addWidget(QLabel("GNSS track (lon/lat, from MTi fusion)"))
        self.gnss_plot = pg.PlotWidget()
        self.gnss_plot.setAspectLocked(True)
        self.gnss_plot.setMinimumHeight(220)
        self.gnss_track = self.gnss_plot.plot([], [], pen=pg.mkPen(120, 160, 255, 150))
        self.gnss_current = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 220, 0, 255))
        self.gnss_plot.addItem(self.gnss_current)
        imu_layout.addWidget(self.gnss_plot)

        imu_layout.addWidget(QLabel("Driven route (OpenStreetMap)"))
        self.map_view = MapView()
        self.map_view.setMinimumHeight(300)
        imu_layout.addWidget(self.map_view)

        splitter.addWidget(_scrollable(imu_panel))
        splitter.setSizes([550, 500, 450])

        # --- events dock ---
        self.events_list = QListWidget()
        self.events_list.itemDoubleClicked.connect(self._seek_to_event)
        events_container = QWidget()
        events_layout = QVBoxLayout(events_container)
        events_layout.addWidget(QLabel("Events (double-click to seek)"))
        events_layout.addWidget(self.events_list)
        splitter.addWidget(events_container)
        splitter.setSizes([550, 500, 450, 200])

        # --- transport bar ---
        transport = QHBoxLayout()
        self.rewind_btn = QPushButton("⏮ Start")
        self.rewind_btn.clicked.connect(lambda: self._seek(0))
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.step_back_btn = QPushButton("◀ Step")
        self.step_back_btn.clicked.connect(lambda: self._seek(self.current_index - 1))
        self.step_fwd_btn = QPushButton("Step ▶")
        self.step_fwd_btn.clicked.connect(lambda: self._seek(self.current_index + 1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider_moved)
        self.time_label = QLabel("00:00:00.000 / 00:00:00.000  (frame 0/0)")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems([f"{s}x" for s in SPEEDS])
        self.speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        # Scrubber + a thin read-only strip of recorded-event markers below
        # it, stacked so they share the same horizontal extent. The strip
        # never replaces the QSlider and never itself drives playback state;
        # clicking a marker just requests a seek via the same path event
        # list double-clicks already use (_seek_to_timestamp).
        slider_column = QVBoxLayout()
        slider_column.setContentsMargins(0, 0, 0, 0)
        slider_column.setSpacing(2)
        slider_column.addWidget(self.slider)
        self.event_marker_strip = EventMarkerStrip()
        self.event_marker_strip.markerClicked.connect(self._seek_to_timestamp)
        slider_column.addWidget(self.event_marker_strip)

        transport.addWidget(self.rewind_btn)
        transport.addWidget(self.step_back_btn)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.step_fwd_btn)
        transport.addLayout(slider_column, stretch=1)
        transport.addWidget(self.time_label)
        transport.addWidget(self.speed_combo)
        outer.addLayout(transport)

    # ------------------------------------------------------------ loading ---
    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output_multi-record/output directory")
        if d:
            self._populate_sessions(Path(d))

    def _populate_sessions(self, root: Path):
        self.dataset_root = root
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "raw" / "recording.svo2").exists()
        )
        for p in dirs:
            self.session_combo.addItem(p.name, p)
        self.session_combo.blockSignals(False)
        if dirs:
            self.session_combo.setCurrentIndex(0)
            self._load_session(dirs[0])

    def _on_session_selected(self, idx: int):
        p = self.session_combo.itemData(idx)
        if p is not None:
            self._load_session(Path(p))

    def _load_session(self, session_dir: Path):
        """Requests that `session_dir` be prepared and become the active
        session -- but does not touch the currently active session at all.

        The actual work (RecordingSession.load, SvoReader/SyncIndex/
        validator/sensor-health construction -- see session_loader.py) runs
        on a dedicated background thread. The session currently on screen
        stays fully usable (playback, scrubbing, dialogs) for as long as
        this one is loading; nothing is torn down until a matching
        SessionLoadResult successfully arrives (_commit_session).
        """
        self._session_generation += 1
        generation = self._session_generation
        self._loading_session_name = session_dir.name
        self._session_load_state = "LOADING"
        self._set_loading_label(stage="starting")

        request = SessionLoadRequest(
            generation=generation, session_dir=session_dir, thresholds=self._sync_thresholds,
        )
        self.requestSessionLoad.emit(request)

    def _set_loading_label(self, stage: str):
        active_note = f"  (current session '{self.session.session_id}' remains active)" if self.session is not None else ""
        self.info_label.setText(f"SESSION: loading {self._loading_session_name}… [{stage}]{active_note}")

    def _on_session_load_progress(self, generation: int, stage: str):
        # Coarse stage labels only -- never a time estimate (see
        # session_loader.py); superseded requests are ignored the same way
        # a superseded finished-result is.
        if generation != self._session_generation:
            return
        self._set_loading_label(stage)

    def _on_session_load_finished(self, result: SessionLoadResult):
        if result.generation != self._session_generation:
            # A newer session request has since been made; this result is
            # obsolete and must never become active (generation-based
            # supersession -- see session_loader.py). Nothing else was ever
            # given a reference to result.svo (SensorWorker only ever sees
            # self.svo, which _commit_session below is the only place that
            # reassigns), so closing it here is safe -- no one else could
            # possibly still be using it.
            logger.debug(
                "discarding superseded session load result gen=%s (current=%s)",
                result.generation, self._session_generation,
            )
            if result.svo is not None:
                result.svo.close()
            return

        if not result.ok:
            self._session_load_state = "FAILED"
            active_note = (
                f"\n\nCurrent session '{self.session.session_id}' remains active."
                if self.session is not None else ""
            )
            QMessageBox.warning(
                self, "Session load failed",
                f"SESSION LOAD FAILED\n\nStage: {result.failed_stage}\n\nReason: {result.error}{active_note}",
            )
            current = f"  (session={self.session.session_id})" if self.session is not None else ""
            self.info_label.setText(f"SESSION: failed to load {result.session_dir.name}: {result.error}{current}")
            return

        self._commit_session(result)

    def _commit_session(self, result: SessionLoadResult):
        """Replaces the active session with an already-fully-prepared one.

        Everything expensive already happened on the loader thread; this
        method only reassigns references and updates widgets, so it's fast
        and safe to run synchronously on the UI thread. Crucially, the old
        session's worker-dependent state (SvoReader, caches, in-flight
        requests) is only torn down *here*, right before the swap -- never
        earlier -- so the previous session stayed fully usable while this
        one was loading.
        """
        self._stop_playback()
        self._heatmap_shape = None

        # Invalidate anything in flight for the old session and wait for
        # SensorWorker to actually finish before touching self.svo below --
        # it may still be mid-decode on the *old* SvoReader's Camera object,
        # which is not safe to close out from under it. This is the same
        # existing protection _load_session always used, just invoked at
        # commit time instead of at request time.
        self._pending_request = None
        self._generation += 1
        self._await_worker_idle()

        # Drop any cached frames/radar samples from the previous session
        # before anything from the new one can be requested -- otherwise a
        # cache hit keyed by a stale id() could, in principle, resurface
        # data read from the old recording (see SensorWorker.reset_caches).
        self.resetCache.emit()

        old_svo = self.svo

        self.session = result.session
        self.svo = result.svo
        self.sync_index = result.sync_index
        self.validation_result = result.validation_result
        self.sensor_health = result.sensor_health
        self._current_health_snapshot = None

        # Causal radar index: precomputed once per session, the same way
        # SyncIndex itself precomputes the NEAREST mapping -- see
        # radar_causal.py. Built from timestamp arrays already sitting in
        # memory (session.radar/session.frames); no disk I/O, no new
        # SyncIndex, no change to the existing NEAREST mapping.
        radar_df = self.session.radar
        self._radar_ts = (
            radar_df["timestamp_ns"].to_numpy()
            if radar_df is not None and not radar_df.empty
            else np.array([], dtype=np.int64)
        )
        frames_df = self.session.frames
        frames_ts = frames_df["timestamp_ns"].to_numpy() if not frames_df.empty else np.array([], dtype=np.int64)
        self._radar_causal_index = causal_indices_for_frames(self._radar_ts, frames_ts)
        self._reset_radar_display()

        if old_svo is not None and old_svo is not self.svo:
            old_svo.close()

        self._session_load_state = "READY"

        f = self.session.frames
        n_frames = len(f)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(n_frames - 1, 0))

        m = self.session.manifest
        sm = self.session.session_manifest
        duration = sm.get("duration_s", 0)
        self.info_label.setText(
            f"session={self.session.session_id}  vehicle={m.get('vehicle') or '-'}  "
            f"camera={m.get('camera')}  {m.get('camera_fps')} fps  {tuple(m.get('resolution', []))}  "
            f"imu={m.get('imu')}  radar={m.get('radar')}  lidar={m.get('lidar')}  "
            f"duration={duration:.1f}s  frames={n_frames}"
        )
        if self.svo is None:
            self.left_label.setText("Camera unavailable")
            self.right_label.setText("Camera unavailable")

        self._plot_imu_series()
        self._plot_zed_imu_series()
        self._plot_gnss_track()
        self._populate_events()
        self._show_lidar_status()
        self._display_validation_result()
        self._populate_event_markers()
        self._setup_timeline_diagnostics()

        self._seek(0)

    def _display_validation_result(self):
        """UI-only: shows the ValidationResult SessionLoader already
        computed (see validator.py) -- no validation runs here."""
        if self.validation_result is None:
            self.integrity_label.setText("validation unavailable")
            return
        rows = self.validation_result.summary_rows()
        self.integrity_label.setText(
            "   ".join(f"{name} {status.value}" for name, status in rows)
        )

    def _populate_event_markers(self):
        frames = self.session.frames if self.session is not None else None
        if self.sensor_health is None or frames is None or frames.empty:
            self.event_marker_strip.set_markers([])
            return

        cam_ts = frames["timestamp_ns"].to_numpy()
        t0, t1 = int(cam_ts[0]), int(cam_ts[-1])
        span = max(t1 - t0, 1)

        markers = []
        for ev in self.sensor_health.drop_events:
            frac = (ev.timestamp_ns - t0) / span
            color = QColor(255, 140, 0) if "RADAR" in ev.event_type else QColor(0, 200, 255)
            markers.append((frac, color, ev.timestamp_ns, f"{ev.event_type}: {ev.description}"))
        self.event_marker_strip.set_markers(markers)

    def _setup_timeline_diagnostics(self):
        """Feeds the TimelineDiagnostics widget the coverage ranges and
        recorded drop events for the newly-committed session, replacing
        whatever the previous session showed atomically (one call, right
        here, as part of the same synchronous commit as every other panel).

        Coverage is read straight from arrays RecordingSession/SvoReader
        already loaded (first/last timestamp of each) -- no new search, no
        DatasetIntegrityValidator invocation, no SyncIndex rebuild.
        """
        frames = self.session.frames if self.session is not None else None
        if frames is None or frames.empty:
            self.timeline_diagnostics.clear()
            return

        cam_ts = frames["timestamp_ns"].to_numpy()
        camera_first_ns, camera_last_ns = int(cam_ts[0]), int(cam_ts[-1])
        coverage: dict[str, tuple[int, int]] = {"Camera": (camera_first_ns, camera_last_ns)}

        def add(name: str, ts_array):
            if ts_array is not None and len(ts_array) > 0:
                coverage[name] = (int(ts_array[0]), int(ts_array[-1]))

        add("Radar", self.session.radar["timestamp_ns"].to_numpy() if not self.session.radar.empty else None)
        add("Lidar", self.session.lidar.timestamps_ns
            if (self.session.lidar is not None and self.session.lidar.num_scans > 0) else None)
        add("MTi-8", self.session.mti.timestamp_ns
            if (self.session.mti is not None and self.session.mti.n > 0) else None)
        if self.svo is not None and self.svo.has_zed_imu:
            add("ZED IMU", self.svo.timestamps_ns)
        # GNSS position lives inside the MTi arrays (see sensor_health.py's
        # GNSS diagnostic) -- its temporal coverage mirrors MTi's whenever
        # MTi has any data at all; fix validity is a per-frame state, not a
        # coverage-range fact, and stays exactly as SensorHealth reports it.
        if "MTi-8" in coverage:
            coverage["GNSS"] = coverage["MTi-8"]

        drop_events = self.sensor_health.drop_events if self.sensor_health is not None else []
        self.timeline_diagnostics.set_session(camera_first_ns, camera_last_ns, coverage, drop_events)

    def _show_timeline_details(self):
        """Expanded, timeline-focused view: coverage + current position/
        sample/delta + recorded events. Deliberately narrower than the
        Dataset Integrity dialog (structural checks stay there) and reuses
        the same SensorHealthSnapshot fields the compact strip/dialog show
        -- nothing here is recalculated.
        """
        if self.session is None:
            QMessageBox.information(self, "Timeline Diagnostics", "No session is loaded.")
            return

        lines = ["SENSOR COVERAGE (first → last recorded timestamp)"]
        for name, (first_ns, last_ns) in self.timeline_diagnostics.coverage().items():
            lines.append(f"  {name}: {first_ns / 1e9:.6f} s → {last_ns / 1e9:.6f} s")
        lines.append("")

        snapshot = self._current_health_snapshot
        if snapshot is not None:
            lines.append(f"CURRENT TIMESTAMP: {snapshot.camera_timestamp_ns / 1e9:.6f} s (frame {snapshot.camera_frame_index})")
            lines.append("")
            for label, diag in snapshot.detail_rows():
                lines.append(label)
                lines.append(f"  Status: {diag.state.display()}")
                if diag.sample_index is not None:
                    lines.append(f"  Sample: {diag.sample_index}")
                if diag.sensor_timestamp_ns is not None:
                    lines.append(f"  Sensor timestamp: {diag.sensor_timestamp_ns / 1e9:.6f} s")
                if diag.delta_ms is not None:
                    sign = "+" if diag.delta_ms >= 0 else ""
                    lines.append(f"  Δt: {sign}{diag.delta_ms:.3f} ms ({diag.sync_class})")
                lines.append("")

        if self.sensor_health is not None and self.sensor_health.drop_events:
            lines.append("RECORDED EVENTS")
            for ev in self.sensor_health.drop_events:
                lines.append(f"  [{ev.event_type}] {ev.description}  @ {ev.timestamp_ns / 1e9:.3f} s")

        QMessageBox.information(self, "Timeline Diagnostics", "\n".join(lines))

    def _update_sensor_health(self):
        """Recomputed synchronously on every seek, from data already in
        memory (see sensor_health.SensorHealthModel.snapshot) -- it never
        waits on SensorWorker, so there is no "generation" to go stale:
        this always reflects exactly `self.current_index`, the same one the
        rest of _update_frame is rendering for.
        """
        if self.sensor_health is None:
            self.sensor_status_label.setText("not available")
            self._current_health_snapshot = None
            self.timeline_diagnostics.set_current(None)
            return
        snapshot = self.sensor_health.snapshot(self.current_index)
        self._current_health_snapshot = snapshot
        self.sensor_status_label.setText(
            "   ".join(f"{name} {state.display()}" for name, state in snapshot.strip_rows())
        )
        self.timeline_diagnostics.set_current(snapshot)

    def _show_sensor_health_details(self):
        snapshot = self._current_health_snapshot
        if snapshot is None:
            QMessageBox.information(self, "Sensor Diagnostics", "Not available for this session.")
            return

        lines = [
            "CURRENT TIMESTAMP",
            f"Camera: {snapshot.camera_timestamp_ns / 1e9:.6f} s (frame {snapshot.camera_frame_index})",
            "",
        ]
        for label, diag in snapshot.detail_rows():
            lines.append(label)
            lines.append(f"  Status: {diag.state.display()}")
            if diag.sample_index is not None:
                lines.append(f"  Sample: {diag.sample_index}")
            if diag.sensor_timestamp_ns is not None:
                lines.append(f"  Sensor timestamp: {diag.sensor_timestamp_ns / 1e9:.6f} s")
            if diag.delta_ms is not None:
                sign = "+" if diag.delta_ms >= 0 else ""
                lines.append(f"  Δt: {sign}{diag.delta_ms:.3f} ms ({diag.sync_class})")
            if diag.detail:
                lines.append(f"  Note: {diag.detail}")
            lines.append("")

        if self.sensor_health is not None and self.sensor_health.drop_events:
            lines.append("RECORDED DROP EVENTS")
            for ev in self.sensor_health.drop_events:
                lines.append(f"  [{ev.event_type}] {ev.description}  @ {ev.timestamp_ns / 1e9:.3f} s")

        QMessageBox.information(self, "Sensor Diagnostics", "\n".join(lines))

    def _show_integrity_details(self):
        if self.validation_result is None:
            QMessageBox.information(self, "Dataset Integrity", "Validation has not run for this session.")
            return
        vr = self.validation_result
        lines = [f"Session: {vr.session_id}   Overall: {vr.overall_status().value}", ""]
        for name, report in vr.all_sections():
            lines.append(f"{name}: {report.status.value}")
            for check in report.checks:
                lines.append(f"    [{check.status.value}] {check.name}: {check.detail}")
            lines.append("")
        QMessageBox.information(self, f"Dataset Integrity — {vr.session_id}", "\n".join(lines))

    def _plot_imu_series(self):
        self.accel_plot.clear()
        self.gyro_plot.clear()
        mti = self.session.mti
        if mti is None or mti.n == 0:
            return
        t0 = mti.timestamp_ns[0]
        t = (mti.timestamp_ns - t0) / 1e9
        colors = ["r", "g", "b"]
        for i, label in enumerate(["x", "y", "z"]):
            self.accel_plot.plot(t, mti.acceleration[:, i], pen=pg.mkPen(colors[i]), name=label)
            self.gyro_plot.plot(t, mti.angular_velocity[:, i], pen=pg.mkPen(colors[i]), name=label)
        self.accel_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y", width=2))
        self.gyro_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y", width=2))
        self.accel_plot.addItem(self.accel_cursor)
        self.gyro_plot.addItem(self.gyro_cursor)

    def _plot_zed_imu_series(self):
        self.zed_accel_plot.clear()
        self.zed_gyro_plot.clear()
        self.zed_accel_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y", width=2))
        self.zed_gyro_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y", width=2))
        self.zed_accel_plot.addItem(self.zed_accel_cursor)
        self.zed_gyro_plot.addItem(self.zed_gyro_cursor)
        if self.svo is None or not self.svo.has_zed_imu:
            title = "ZED onboard IMU not available in this recording" if self.svo is not None else ""
            self.zed_accel_plot.setTitle(title)
            self.zed_gyro_plot.setTitle(title)
            return
        self.zed_accel_plot.setTitle("")
        self.zed_gyro_plot.setTitle("")
        t0 = self.svo.timestamps_ns[0]
        t = (self.svo.timestamps_ns - t0) / 1e9
        colors = ["r", "g", "b"]
        for i in range(3):
            self.zed_accel_plot.plot(t, self.svo.zed_accel[:, i], pen=pg.mkPen(colors[i]))
            self.zed_gyro_plot.plot(t, self.svo.zed_gyro[:, i], pen=pg.mkPen(colors[i]))

    def _reset_radar_display(self):
        """Drops the previous session's 2D/3D radar visuals before the new
        session's first frame is rendered -- same reasoning as
        Lidar3DView.clear() in _show_lidar_status(). Neither widget is
        destroyed; only their displayed data/header text is reset."""
        self.points_scatter.setData([], [])
        self.points_plot.setTitle("")
        self.heatmap_view.clear()
        self.heatmap_view.ui.histogram.hide()
        self._heatmap_shape = None
        self.radar_3d_view.clear()
        self._last_radar_selection = (None, None, False, False)
        self.radar_header_label.setText("RADAR\n(loading…)")

    def _on_radar_view_mode_changed(self, mode: str):
        """Presentation-only: switches which already-populated widget is
        shown. Never touches session/SyncIndex/radar data, never re-seeks,
        never re-dispatches to the worker -- see design brief §2/§5/§18."""
        if mode not in RADAR_VIEW_MODES or mode == self._radar_view_mode:
            return
        self._radar_view_mode = mode
        self.radar_view_stack.setCurrentIndex(RADAR_VIEW_MODES.index(mode))
        self._update_radar_header_from_current_state()

    def _on_radar_temporal_mode_changed(self, mode: str):
        """Switches the display-only temporal policy (see radar_causal.py)
        and re-resolves *only* the radar sample for the current frame --
        the camera frame, playback position, and SensorHealthModel are
        completely untouched (design brief §16-§19)."""
        if mode not in RADAR_TEMPORAL_MODES or mode == self._radar_temporal_mode:
            return
        self._radar_temporal_mode = mode
        if self.session is None or self.session.frames.empty:
            return
        ts = int(self.session.frames.iloc[self.current_index]["timestamp_ns"])
        self._dispatch_async(ts, include_camera=False)

    def _resolve_radar_selection(self, ts: int):
        """Returns (radar_row_or_None, delta_ns_or_None, is_future, no_causal_yet).

        NEAREST reuses SyncIndex.radar_for() verbatim -- the existing
        nearest-sample lookup is never changed (design brief §8/§17).
        CAUSAL indexes the precomputed `_radar_causal_index` array built by
        radar_causal.causal_indices_for_frames() at session-commit time --
        a display-only restriction against future timestamps, not a new
        synchronization system (design brief §9/§17).
        """
        if self.session is None or self.sync_index is None or self.session.radar.empty:
            return None, None, False, False

        if self._radar_temporal_mode == "CAUSAL":
            if self.current_index >= len(self._radar_causal_index):
                return None, None, False, True
            pos = int(self._radar_causal_index[self.current_index])
            if pos == MISSING:
                return None, None, False, True  # "no causal sample yet" -- never fall back to nearest/future
            row = self.session.radar.iloc[pos]
            delta_ns = int(row["timestamp_ns"]) - int(ts)
            return row, delta_ns, False, False

        # NEAREST -- byte-identical to the pre-existing behavior.
        pos = self.sync_index.radar_for(self.current_index)
        if pos == MISSING:
            return None, None, False, False
        row = self.session.radar.iloc[pos]
        delta_ns = int(row["timestamp_ns"]) - int(ts)
        return row, delta_ns, delta_ns > 0, False

    def _update_radar_header_from_current_state(self):
        """Re-renders the header from whatever radar selection was last
        resolved, without performing a new lookup -- used when only the
        2D/3D toggle changes (design brief §15/§18), so the "2D"/"3D" line
        itself still updates even though the underlying sample doesn't."""
        row, delta_ns, is_future, no_causal_yet = self._last_radar_selection
        self._render_radar_header(row, delta_ns, is_future, no_causal_yet)

    def _update_radar_header(self, row, delta_ns, is_future: bool, no_causal_yet: bool):
        self._last_radar_selection = (row, delta_ns, is_future, no_causal_yet)
        self._render_radar_header(row, delta_ns, is_future, no_causal_yet)

    def _render_radar_header(self, row, delta_ns, is_future: bool, no_causal_yet: bool):
        lines = ["RADAR", self._radar_view_mode, self._radar_temporal_mode]
        if row is None:
            lines.append("No causal sample yet" if no_causal_yet else "No data")
        else:
            sample_id = row.get("frame_id") if hasattr(row, "get") else None
            lines.append(f"Sample {sample_id}" if sample_id is not None else "Sample (unlabeled)")
            lines.append(f"t = {int(row['timestamp_ns']) / 1e9:.6f} s")
            sign = "+" if delta_ns >= 0 else ""
            lines.append(f"Δt = {sign}{delta_ns / 1e6:.1f} ms")
            if is_future:
                lines.append(
                    f"⚠ Future sample: recorded {delta_ns / 1e6:.1f} ms after the current "
                    "camera time (diagnostic, not an error)"
                )
        health_state = None
        if self._current_health_snapshot is not None:
            health_state = self._current_health_snapshot.radar.state.display()
        lines.append(health_state if health_state else "—")
        self.radar_header_label.setText("\n".join(lines))

    def _clear_radar_display(self, no_causal_yet: bool):
        """No radar row was selected at all for this frame (design brief
        §10) -- clears both widgets and the heatmap honestly rather than
        leaving the previous frame's data on screen. No worker round-trip
        happens for this case, so this must run synchronously."""
        message = "No causal sample yet" if no_causal_yet else "no radar sample for this frame"
        self.points_scatter.setData([], [])
        self.points_plot.setTitle(f"radar: {message}")
        self.heatmap_view.clear()
        self.heatmap_view.ui.histogram.hide()
        self._heatmap_shape = None
        self.radar_3d_view.clear()

    def _show_lidar_status(self):
        # Drop the previous session's cloud/one-time-fit state before
        # showing anything for the new one (see Lidar3DView.clear()).
        self.lidar_view.clear()
        if self.session.lidar is not None:
            self.lidar_view.set_session_info(f"{self.session.lidar.num_scans} scans loaded")
        elif self.session.lidar_error:
            self.lidar_view.show_missing(f"Lidar unavailable: {self.session.lidar_error}")
        else:
            self.lidar_view.show_missing("No lidar.pcap/lidar_0.json found for this session")

    def _plot_gnss_track(self):
        mti = self.session.mti
        if mti is None or mti.n == 0:
            self.gnss_track.setData([], [])
            self.map_view.set_track([])
            return
        lon, lat = mti.longitude, mti.latitude
        valid = np.isfinite(lon) & np.isfinite(lat) & (lon != 0) & (lat != 0)
        self.gnss_track.setData(lon[valid], lat[valid])
        self.gnss_plot.setTitle("" if valid.any() else "no GNSS fix in this recording")
        self.map_view.set_track(list(zip(lat[valid].tolist(), lon[valid].tolist())))

    def _populate_events(self):
        self.events_list.clear()
        ev = self.session.events
        if ev.empty:
            return
        t0 = self.session.frames["timestamp_ns"].iloc[0] if len(self.session.frames) else ev["timestamp_ns"].iloc[0]
        for _, row in ev.iterrows():
            rel = ns_to_str(int(row["timestamp_ns"]) - int(t0))
            item_text = f"[{rel}] {row['event_type']}: {row['description']}"
            self.events_list.addItem(item_text)
            self.events_list.item(self.events_list.count() - 1).setData(Qt.UserRole, int(row["timestamp_ns"]))

    def _seek_to_event(self, item):
        ts = item.data(Qt.UserRole)
        if ts is not None:
            self._seek_to_timestamp(ts)

    def _seek_to_timestamp(self, timestamp_ns: int):
        """Seeks to the nearest camera frame at or after `timestamp_ns`.

        The one existing event-to-camera-frame seeking rule in this
        application, shared by the events list (double-click) and the new
        recorded-event timeline markers (click) -- neither changes it.
        """
        if self.session is None or self.session.frames.empty:
            return
        frames_ts = self.session.frames["timestamp_ns"].to_numpy()
        idx = int(np.searchsorted(frames_ts, timestamp_ns))
        idx = max(0, min(idx, len(frames_ts) - 1))
        self._seek(idx)

    # ------------------------------------------------------------ playback ---
    def _toggle_play(self):
        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self.session is None or self.session.frames.empty:
            return
        fps = float(self.session.manifest.get("camera_fps", 30.0)) or 30.0
        interval_ms = max(1, int(1000.0 / fps / self.speed))
        self.timer.start(interval_ms)
        self.playing = True
        self.play_btn.setText("⏸ Pause")

    def _stop_playback(self):
        self.timer.stop()
        self.playing = False
        self.play_btn.setText("▶ Play")

    def _on_speed_changed(self, idx: int):
        self.speed = SPEEDS[idx]
        if self.playing:
            self._stop_playback()
            self._start_playback()

    def _advance(self):
        n = len(self.session.frames) if self.session is not None else 0
        if self.current_index + 1 >= n:
            self._stop_playback()
            return
        self._seek(self.current_index + 1)

    def _on_slider_moved(self, value: int):
        if value != self.current_index:
            self._seek(value)

    def _seek(self, index: int):
        if self.session is None or self.session.frames.empty:
            return
        n = len(self.session.frames)
        index = max(0, min(index, n - 1))
        self.current_index = index
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self._update_frame()

    # -------------------------------------------------------------- render ---
    def _update_frame(self):
        if self.session is None or self.session.frames.empty:
            return
        row = self.session.frames.iloc[self.current_index]
        ts = int(row["timestamp_ns"])
        t0 = int(self.session.frames["timestamp_ns"].iloc[0])
        t_end = int(self.session.frames["timestamp_ns"].iloc[-1])
        self.time_label.setText(
            f"{ns_to_str(ts - t0)} / {ns_to_str(t_end - t0)}  (frame {self.current_index + 1}/{len(self.session.frames)})"
        )

        # The IMU/GNSS cursors, sensor-health diagnostics, and lidar all
        # only touch data already fully loaded in memory (see
        # SYSTEM_FUNCTIONS.md, sensor_health.py) -- no disk I/O happens
        # here, so these stay synchronous on the UI thread. Sensor health
        # runs first so _render_lidar can reuse its already-computed LiDAR
        # SensorDiagnostic (delta/state) instead of recalculating it.
        self._update_sensor_health()
        self._render_lidar(ts)
        self._render_imu_cursor(ts, t0)

        # Camera decode and radar file reads are the actual blocking disk/
        # CPU work per frame; dispatch those to the background worker.
        self._dispatch_async(ts)

    @staticmethod
    def _set_pixmap(label: QLabel, img: np.ndarray):
        h, w, _ = img.shape
        qimg = QImage(np.ascontiguousarray(img).data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        label.setPixmap(pix)

    # ------------------------------------------------------ async dispatch ---
    def _dispatch_async(self, ts: int, include_camera: bool = True):
        """Build a request for the current frame using the precomputed sync
        index, then hand the actual I/O to the worker.

        The index lookup is now an O(1) array read (see sync_index.py) built
        once per session/SvoReader instead of a fresh search on every seek;
        it selects exactly the same sample RecordingSession.nearest_radar_row
        / SvoReader.nearest_frame_index would. Only the resulting SVO decode
        and radar file reads move to the worker, same as before.

        `include_camera=False` is used when only the radar temporal mode
        changed (see _on_radar_temporal_mode_changed): the request carries
        no SvoReader/svo_index, so worker.py's process() skips the camera
        decode entirely and only (re)reads the newly-selected radar sample
        -- no new worker, no change to the camera/lidar/IMU rendering path.
        """
        self._generation += 1
        generation = self._generation

        svo_index = None
        if include_camera and self.svo is not None and self.sync_index is not None:
            mapped = self.sync_index.svo_for(self.current_index)
            svo_index = mapped if mapped != MISSING else None

        radar_row, radar_delta_ns, is_future, no_causal_yet = self._resolve_radar_selection(ts)
        self._update_radar_header(radar_row, radar_delta_ns, is_future, no_causal_yet)
        if radar_row is None:
            # No worker round-trip will happen for radar (see below), so the
            # "no data"/"no causal sample yet" display must be applied here,
            # synchronously -- never left showing a previous frame's data.
            self._clear_radar_display(no_causal_yet)

        request = FrameRequest(
            generation=generation,
            svo=self.svo if include_camera else None,
            svo_index=svo_index,
            session=self.session,
            radar_row=radar_row,
            radar_delta_ns=radar_delta_ns,
        )

        if self._worker_busy:
            # Coalesce: only the newest request survives. Bounded to at most
            # one in-flight + one pending, so rapid scrubbing never queues
            # unbounded work.
            logger.debug("worker busy, coalescing request gen=%s", generation)
            self._pending_request = request
            return

        self._send_to_worker(request)

    def _send_to_worker(self, request: FrameRequest):
        self._worker_busy = True
        self.requestFrame.emit(request)

    def _await_worker_idle(self, timeout_ms: int = WORKER_IDLE_TIMEOUT_MS):
        """Block briefly until the worker finishes its current job, if any.

        Only called on session switch (an explicit, infrequent user action),
        never during normal playback or scrubbing -- it exists solely so we
        never close/replace a SvoReader the worker thread might still be
        using.
        """
        if not self._worker_busy:
            return
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        self._worker.finished.connect(loop.quit)
        timer.start(timeout_ms)
        loop.exec()
        try:
            self._worker.finished.disconnect(loop.quit)
        except (RuntimeError, TypeError):
            pass
        if self._worker_busy:
            logger.warning("worker did not go idle within %d ms during session switch", timeout_ms)
        self._worker_busy = False
        self._pending_request = None

    def _on_camera_ready(self, generation: int, payload: dict):
        if generation != self._generation:
            return  # stale: user has already scrubbed elsewhere
        self._set_pixmap(self.left_label, payload["left"])
        self._set_pixmap(self.right_label, payload["right"])

    def _on_radar_ready(self, generation: int, payload: dict):
        if generation != self._generation:
            return  # stale: user has already scrubbed elsewhere
        heatmap = payload["heatmap"]
        if heatmap is not None:
            first = self._heatmap_shape is None
            self.heatmap_view.ui.histogram.show()  # may have been hidden by _clear_radar_display
            self.heatmap_view.setImage(heatmap.astype(np.float32), autoLevels=True, autoRange=first)
            self._heatmap_shape = heatmap.shape
        points = payload["points"]
        # Both the 2D and 3D widgets are updated from this same payload,
        # regardless of which one is currently visible -- so switching the
        # view toggle never causes an independent lookup (design brief §6).
        if points:
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            self.points_scatter.setData(xs, ys)
            zs = [p.get("z", float("nan")) for p in points]
            xyz = np.array(list(zip(xs, ys, zs)), dtype=np.float32)
            self.radar_3d_view.set_points(xyz)
        else:
            self.points_scatter.setData([], [])
            self.radar_3d_view.clear()
        delta_ns = payload["delta_ns"]
        delta_str = f"{delta_ns / 1e6:.1f} ms" if delta_ns is not None else "n/a"
        self.points_plot.setTitle(f"radar Δt={delta_str}, {len(points)} detections")

    def _on_worker_error(self, generation: int, kind: str, message: str):
        logger.warning("worker error gen=%s kind=%s: %s", generation, kind, message)
        if generation != self._generation:
            return  # stale: don't show an error for a frame we've left
        if kind == "camera":
            self.left_label.setText(f"Camera: frame decode failed\n{message}")
            self.right_label.setText(f"Camera: frame decode failed\n{message}")
        elif kind == "radar":
            self.points_plot.setTitle(f"radar unavailable: {message}")

    def _on_worker_finished(self, generation: int):
        self._worker_busy = False
        if self._pending_request is not None:
            pending = self._pending_request
            self._pending_request = None
            self._send_to_worker(pending)

    def _render_lidar(self, ts: int):
        lidar = self.session.lidar
        if lidar is None or lidar.num_scans == 0 or self.sync_index is None:
            self.lidar_view.show_missing("No LiDAR data for this session")
            return
        idx = self.sync_index.lidar_for(self.current_index)
        if idx == MISSING:
            self.lidar_view.show_missing("No LiDAR data for this frame")
            return
        pts = lidar.points_at(idx)
        # The existing sensor-health snapshot already located this same
        # sample via SyncIndex and computed its delta/state -- reused here
        # verbatim for the header, never recalculated (see lidar_3d_view.py).
        diagnostic = self._current_health_snapshot.lidar if self._current_health_snapshot is not None else None
        self.lidar_view.set_points(pts, scan_index=idx, timestamp_ns=int(lidar.timestamps_ns[idx]), diagnostic=diagnostic)

    def _render_imu_cursor(self, ts: int, t0: int):
        mti = self.session.mti
        if mti is not None and mti.n > 0 and self.sync_index is not None:
            idx = self.sync_index.mti_for(self.current_index)
            if idx != MISSING:
                t_rel = (int(mti.timestamp_ns[idx]) - t0) / 1e9
                self.accel_cursor.setPos(t_rel)
                self.gyro_cursor.setPos(t_rel)
                lon, lat = mti.longitude[idx], mti.latitude[idx]
                if np.isfinite(lon) and np.isfinite(lat) and lon != 0 and lat != 0:
                    self.gnss_current.setData([lon], [lat])
                    self.map_view.set_position(float(lat), float(lon))

        if self.svo is not None and self.svo.has_zed_imu and self.sync_index is not None:
            # Same mapping already computed for the camera decode request in
            # _dispatch_async (same ts, same SvoReader) -- reused here rather
            # than searched again.
            zed_idx = self.sync_index.svo_for(self.current_index)
            if zed_idx != MISSING:
                zed_t_rel = (int(self.svo.timestamps_ns[zed_idx]) - t0) / 1e9
                self.zed_accel_cursor.setPos(zed_t_rel)
                self.zed_gyro_cursor.setPos(zed_t_rel)

    def closeEvent(self, event):
        # Invalidate any in-flight/queued session load FIRST, before doing
        # anything that spins a nested Qt event loop below. _await_worker_idle
        # runs a nested QEventLoop while waiting for SensorWorker; if a
        # SessionLoader.finished result for the *current* generation arrived
        # during that nested loop, it would still be accepted and trigger a
        # full _commit_session() (including a fresh requestFrame.emit() to a
        # worker thread that's mid-shutdown) in the middle of window close.
        # Bumping the generation here first means any such result is always
        # recognized as stale and discarded (its SvoReader closed) instead.
        self._session_generation += 1

        self._stop_playback()
        self._pending_request = None
        self._generation += 1
        self._await_worker_idle()
        self._worker_thread.quit()
        self._worker_thread.wait(2000)

        self._session_loader_thread.quit()
        self._session_loader_thread.wait(2000)

        if self.svo is not None:
            self.svo.close()
        super().closeEvent(event)
