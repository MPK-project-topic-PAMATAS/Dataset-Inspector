"""Standalone TI AWR1843BOOST radar diagnostic dashboard.

This is a SEPARATE, independently-launchable PySide6 application. It is
NOT integrated into the main Dataset Inspector UI (main_window.py) and
must never be imported by it. Run it directly:

    python tests/radar_dashboard.py [dataset_root] [session_dir]

It shows BOTH:
  (A) the radar's actual TI mmWave SDK configuration, as literally read
      from the Data-Collection-GCS project (never guessed -- see
      radar_config.py), and
  (B) the actual recorded radar measurements from an existing recording
      under the Dataset Inspector's output_multi-record dataset,
so the two can be compared side by side (a Config-vs-Actual table).

Hard rules enforced throughout this file:
  * Never writes to output_multi-record/ or to Data-Collection-GCS.
    Every access to those trees is a read (RecordingSession.load,
    session.load_radar_points/load_radar_heatmap, radar_config.py's
    text-only parsing).
  * Never fuses radar data with camera/LiDAR/GNSS data. The only
    non-radar value shown anywhere is a single, clearly labeled
    "external reference" vehicle-speed figure from the MTi IMU, which
    is never combined with or substituted for the radar's own target
    velocities.
  * Never fabricates a field that isn't actually present in the recorded
    data (e.g. no SNR/track-ID columns are invented for target objects).
  * Dropped-frame counts are shown only from explicit recorded event
    metadata (RADAR_DROPPED_FRAMES), verbatim, never inferred from
    timestamp gaps. A separate, clearly non-drop-labeled timestamp
    interval diagnostic is always available regardless.

Live hardware mode: this dashboard can also connect directly to a
physically-connected TI AWR1843BOOST (its two ttyACM ports, auto-detected)
to show what would be recorded right now, before starting an actual
recording. This reuses Data-Collection-GCS's own, already-tested
`radar.receiver.RadarReceiver` driver by IMPORTING it (never copying,
never modifying) -- it is the authoritative, hardware-validated serial/TLV
implementation, and reimplementing that protocol here would only add risk.
Importing it has no side effects: it only opens serial ports when its
`.start()` method is explicitly called (from the "Connect Live Radar"
button), and `.stop()` closes them again on disconnect or window close.
The live sensor configuration sent is whatever is currently checked into
`radar/receiver.py` (`_SENSOR_CONFIG_CMDS`) -- at the time this was
written that is a 3TX/4RX ("channelCfg 15 7 0") configuration with the
range-Doppler heatmap output disabled (see the Notes/Conflicts tab).
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from session import RecordingSession  # noqa: E402

import radar_config  # noqa: E402

DEFAULT_DATASET_ROOT = Path("/media/vgtu/ssd11/vgtu/Projects/Dataset-Inspector/output_multi-record")
RECENT_INTERVAL_HISTORY = 200
SPEED_CHOICES = [0.25, 0.5, 1.0, 2.0, 4.0]

_NA = "N/A / Not available"


def _fmt(value, unit="", precision=3):
    if value is None:
        return _NA
    if isinstance(value, float):
        return f"{value:.{precision}f}{(' ' + unit) if unit else ''}"
    return f"{value}{(' ' + unit) if unit else ''}"


def _array_stats(arr) -> dict:
    """Read-only numpy array diagnostics. Never mutates `arr`."""
    if arr is None:
        return {"available": False}
    arr = np.asarray(arr)
    finite_mask = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, dtype=bool)
    finite_vals = arr[finite_mask]
    return {
        "available": True,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(finite_vals.min()) if finite_vals.size else None,
        "max": float(finite_vals.max()) if finite_vals.size else None,
        "mean": float(finite_vals.mean()) if finite_vals.size else None,
        "finite_count": int(finite_mask.sum()),
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "total_count": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# 3D detected-object view.
#
# The AWR1843BOOST in its current (3TX/4RX) configuration genuinely reports
# elevation (z) -- confirmed live against the physically-connected board,
# where z varies meaningfully rather than being flat. A 2D top-down (x,y)
# view would silently discard that axis, so detections are rendered in 3D.
#
# The widget itself now lives in radar_3d_view.py at the project root, so
# both this standalone dashboard and the main Dataset Inspector
# (main_window.py) render radar point clouds through the exact same code --
# see radar_3d_view.py's module docstring.
# --------------------------------------------------------------------------- #

from radar_3d_view import RadarPointCloud3DView  # noqa: E402


# --------------------------------------------------------------------------- #
# Read-only data model for one session's radar stream.
# --------------------------------------------------------------------------- #

class RadarSessionData:
    """Wraps the radar-relevant slice of an already-loaded RecordingSession.
    Read-only: never mutates session.radar / session.events / session.mti.
    """

    def __init__(self, session: RecordingSession):
        self.session = session
        radar_df = session.radar
        if radar_df is None or radar_df.empty:
            self.radar_df = pd.DataFrame()
        else:
            self.radar_df = radar_df.sort_values("timestamp_ns").reset_index(drop=True)
        self.rate_stats = self._compute_rate_stats()
        self.drop_events = self._compute_drop_events()

    @property
    def n_frames(self) -> int:
        return len(self.radar_df)

    def _compute_rate_stats(self):
        if len(self.radar_df) < 2:
            return None
        ts = self.radar_df["timestamp_ns"].to_numpy()
        dt_ms = np.diff(ts).astype(np.float64) / 1e6
        dt_ms = dt_ms[dt_ms > 0]
        if dt_ms.size == 0:
            return None
        hz = 1000.0 / dt_ms
        return {
            "n_intervals": int(dt_ms.size),
            "mean_hz": float(np.mean(hz)),
            "median_hz": float(np.median(hz)),
            "min_hz": float(np.min(hz)),
            "max_hz": float(np.max(hz)),
            "median_interval_ms": float(np.median(dt_ms)),
            "p95_interval_ms": float(np.percentile(dt_ms, 95)),
        }

    def _compute_drop_events(self):
        events = self.session.events
        if events is None or events.empty or "event_type" not in events.columns:
            return []
        mask = events["event_type"].astype(str).str.contains("RADAR_DROPPED_FRAMES", case=False, na=False)
        rows = events[mask].sort_values("timestamp_ns")
        return [
            {"timestamp_ns": int(r["timestamp_ns"]), "description": str(r.get("description", ""))}
            for _, r in rows.iterrows()
        ]

    def row_at(self, index: int):
        if self.radar_df.empty or not (0 <= index < len(self.radar_df)):
            return None
        return self.radar_df.iloc[index]

    def points_at(self, index: int) -> tuple[list, str | None]:
        """Returns (points, error). Never raises; never mutates recording files."""
        row = self.row_at(index)
        if row is None:
            return [], "NO DATA: no radar frame at this index"
        try:
            pts = self.session.load_radar_points(row)
            return pts, None
        except (OSError, json.JSONDecodeError, KeyError) as e:
            return [], f"INVALID DATA: could not read points file ({e})"

    def heatmap_at(self, index: int) -> tuple[np.ndarray | None, str | None]:
        row = self.row_at(index)
        if row is None:
            return None, "NO DATA: no radar frame at this index"
        try:
            hm = self.session.load_radar_heatmap(row)
        except (OSError, ValueError) as e:
            return None, f"INVALID DATA: could not read heatmap file ({e})"
        if hm is None:
            return None, "OPTIONAL DATA UNAVAILABLE: no heatmap recorded for this frame"
        return hm, None

    def external_reference_speed_mps(self, timestamp_ns: int) -> float | None:
        """MTi fused velocity magnitude -- an external reference ONLY.
        Never used as, or merged with, the radar's own target velocity."""
        mti = self.session.mti
        if mti is None or mti.n == 0:
            return None
        idx = mti.nearest_index(timestamp_ns)
        v = mti.velocity[idx]
        if v is None or len(v) == 0 or not np.all(np.isfinite(v)):
            return None
        return float(np.linalg.norm(v))


# --------------------------------------------------------------------------- #
# Playback controller -- timed strictly by actual recorded timestamp deltas.
# --------------------------------------------------------------------------- #

class RadarPlayback(QObject):
    frameChanged = Signal(int)
    playingChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timestamps_ns = np.array([], dtype=np.int64)
        self.index = 0
        self.speed = 1.0
        self.playing = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

    def set_timestamps(self, timestamps_ns: np.ndarray):
        self.pause()
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self._set_index(0)

    @property
    def n(self) -> int:
        return len(self.timestamps_ns)

    def play(self):
        if self.n <= 1 or self.index >= self.n - 1:
            return
        self.playing = True
        self.playingChanged.emit(True)
        self._schedule_next()

    def pause(self):
        was_playing = self.playing
        self.playing = False
        self._timer.stop()
        if was_playing:
            self.playingChanged.emit(False)

    def step(self, delta: int = 1):
        self.pause()
        self._set_index(self.index + delta)

    def reset(self):
        self.pause()
        self._set_index(0)

    def set_speed(self, speed: float):
        self.speed = float(speed)

    def _schedule_next(self):
        if self.index >= self.n - 1:
            self.pause()
            return
        dt_ns = int(self.timestamps_ns[self.index + 1]) - int(self.timestamps_ns[self.index])
        interval_ms = max(1, int((dt_ns / 1e6) / max(self.speed, 1e-6)))
        self._timer.start(interval_ms)

    def _advance(self):
        if not self.playing:
            return
        self._set_index(self.index + 1)
        self._schedule_next()

    def _set_index(self, idx: int):
        idx = 0 if self.n == 0 else max(0, min(self.n - 1, idx))
        self.index = idx
        self.frameChanged.emit(idx)


# --------------------------------------------------------------------------- #
# Live-hardware connection (real, physically-connected AWR1843BOOST).
# --------------------------------------------------------------------------- #

class LiveRadarWorker(QObject):
    """Runs RadarReceiver.start() off the Qt UI thread (it blocks for the
    duration of the CLI config handshake, up to a few seconds). Once
    started, RadarReceiver keeps its own background threads running and
    is safe to poll (get_frame/dropped_frames/rate) from any thread."""

    started = Signal()
    errorOccurred = Signal(str)

    def __init__(self, receiver):
        super().__init__()
        self._receiver = receiver

    def run(self):
        try:
            self._receiver.start()
        except Exception as e:  # noqa: BLE001 -- must never crash the app
            self.errorOccurred.emit(str(e))
            return
        self.started.emit()


def _import_radar_receiver():
    """Imports Data-Collection-GCS's radar.receiver module by path. Read-only:
    this never writes to Data-Collection-GCS, it only loads its (unmodified)
    source as a library."""
    root = str(radar_config.DATA_COLLECTION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from radar.receiver import RadarReceiver  # noqa: PLC0415
    return RadarReceiver


# --------------------------------------------------------------------------- #
# Main dashboard window.
# --------------------------------------------------------------------------- #

class RadarDashboard(QMainWindow):
    def __init__(self, dataset_root: Path = DEFAULT_DATASET_ROOT, session_dir: Path | None = None):
        super().__init__()
        self.setWindowTitle("Radar Diagnostic Dashboard (standalone, read-only)")
        self.dataset_root = Path(dataset_root)
        self.data: RadarSessionData | None = None
        self.config_report: radar_config.RadarConfigReport = radar_config.load_radar_config()
        self._interval_history: deque = deque(maxlen=RECENT_INTERVAL_HISTORY)
        self._last_ts_for_history = None
        self.playback = RadarPlayback(self)
        self.playback.frameChanged.connect(self._on_frame_changed)
        self.playback.playingChanged.connect(self._on_playing_changed)

        self._live_receiver = None
        self._live_thread = None
        self._live_worker = None
        self._last_live_frame_id = None
        self._last_live_ts_ns = None
        self._live_interval_history = deque(maxlen=RECENT_INTERVAL_HISTORY)
        self._live_poll_timer = QTimer(self)
        self._live_poll_timer.setInterval(50)
        self._live_poll_timer.timeout.connect(self._poll_live_frame)

        self._build_ui()

        if session_dir is not None:
            self.load_session(Path(session_dir))
        else:
            self._populate_session_selector()

    # -- session discovery / loading --------------------------------- #

    def _discover_sessions(self) -> list[Path]:
        out_dir = self.dataset_root / "output"
        if not out_dir.exists():
            return []
        return sorted(p for p in out_dir.iterdir() if p.is_dir())

    def _populate_session_selector(self):
        self.session_combo.clear()
        sessions = self._discover_sessions()
        if not sessions:
            self.session_combo.addItem("(no sessions found)")
            self._show_status("NO DATA: no sessions found under " + str(self.dataset_root))
            return
        for s in sessions:
            self.session_combo.addItem(s.name, userData=s)

    def _on_session_selected(self, idx: int):
        path = self.session_combo.itemData(idx)
        if path is not None:
            self.load_session(Path(path))

    def load_session(self, session_dir: Path):
        try:
            session = RecordingSession.load(session_dir)
        except Exception as e:  # noqa: BLE001 -- must never crash the dashboard
            QMessageBox.critical(self, "Session load failed", f"ERROR loading session:\n{e}")
            self._show_status(f"ERROR: could not load session {session_dir}: {e}")
            return

        self.data = RadarSessionData(session)
        self._interval_history.clear()
        self._last_ts_for_history = None

        if self.data.n_frames == 0:
            self._show_status(f"NO DATA: session '{session.session_id}' has no recorded radar frames")
            self.playback.set_timestamps(np.array([], dtype=np.int64))
        else:
            self._show_status(None)
            self.playback.set_timestamps(self.data.radar_df["timestamp_ns"].to_numpy())

        self._refresh_config_panel()
        self._on_frame_changed(self.playback.index)

    # -- UI construction ------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self._on_session_selected)
        selector_row.addWidget(self.session_combo)
        selector_row.addStretch(1)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #b00; font-weight: bold;")
        selector_row.addWidget(self.status_label)
        root.addLayout(selector_row)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([340, 480, 340])

        root.addWidget(self._build_transport_bar())
        root.addWidget(self._build_config_panel())

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        status_box = QGroupBox("RADAR STATUS")
        self.status_form = QFormLayout(status_box)
        self._status_fields = {}
        for name in ["Radar", "Status", "Frame ID", "Timestamp (ns)", "Frame interval (ms)",
                     "Measured rate (mean)", "Measured rate (median)", "Configured rate",
                     "Effective rate note", "Dropped frames (recorded)"]:
            lbl = QLabel(_NA)
            lbl.setWordWrap(True)
            self._status_fields[name] = lbl
            self.status_form.addRow(name + ":", lbl)
        layout.addWidget(status_box)

        target_box = QGroupBox("TARGET / OBJECT INFORMATION (radar-only, no fusion)")
        target_layout = QVBoxLayout(target_box)
        self.target_table = QTableWidget(0, 4)
        self.target_table.setHorizontalHeaderLabels(["#", "x (m)", "y (m)", "velocity (m/s)"])
        self.target_table.setEditTriggers(QTableWidget.NoEditTriggers)
        target_layout.addWidget(self.target_table)
        self.target_note = QLabel("")
        self.target_note.setWordWrap(True)
        target_layout.addWidget(self.target_note)
        layout.addWidget(target_box)

        speed_box = QGroupBox("SPEED")
        speed_layout = QFormLayout(speed_box)
        self.radar_speed_label = QLabel(_NA)
        self.radar_speed_label.setStyleSheet("font-weight: bold;")
        speed_layout.addRow("Radar target speed (radial, from detections):", self.radar_speed_label)
        self.external_speed_label = QLabel(_NA)
        self.external_speed_label.setStyleSheet("color: #555;")
        speed_layout.addRow("Vehicle speed (EXTERNAL REFERENCE -- MTi, NOT radar):", self.external_speed_label)
        note = QLabel("These two values are never fused or substituted for each other.")
        note.setStyleSheet("font-style: italic; color: #777;")
        speed_layout.addRow(note)
        layout.addWidget(speed_box)

        drop_box = QGroupBox("DROPPED-FRAME / TIMESTAMP DIAGNOSTICS")
        drop_layout = QVBoxLayout(drop_box)
        self.drop_event_label = QLabel(_NA)
        self.drop_event_label.setWordWrap(True)
        drop_layout.addWidget(QLabel("Recorded drop event (explicit metadata only):"))
        drop_layout.addWidget(self.drop_event_label)
        self.interval_diag_label = QLabel(_NA)
        self.interval_diag_label.setWordWrap(True)
        drop_layout.addWidget(QLabel("Timestamp interval diagnostic (NOT a drop count):"))
        drop_layout.addWidget(self.interval_diag_label)
        layout.addWidget(drop_box)

        live_box = QGroupBox("LIVE RADAR (physically-connected hardware)")
        live_layout = QVBoxLayout(live_box)
        self.live_button = QPushButton("Connect Live Radar (3TX/4RX)")
        self.live_button.clicked.connect(self._toggle_live_radar)
        live_layout.addWidget(self.live_button)
        self.live_status_label = QLabel("Not connected.")
        self.live_status_label.setWordWrap(True)
        live_layout.addWidget(self.live_status_label)
        self.live_elevation_label = QLabel(_NA)
        self.live_elevation_label.setWordWrap(True)
        live_layout.addWidget(QLabel("Elevation (z) diagnostic -- tests whether the "
                                      "connected board's 3TX/4RX config actually yields "
                                      "non-zero elevation:"))
        live_layout.addWidget(self.live_elevation_label)
        layout.addWidget(live_box)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("Range-Doppler heatmap (recorded only -- never regenerated from raw ADC):"))
        self.heatmap_view = pg.ImageView()
        self.heatmap_view.ui.roiBtn.hide()
        self.heatmap_view.ui.menuBtn.hide()
        layout.addWidget(self.heatmap_view, 2)
        self.heatmap_status = QLabel("")
        layout.addWidget(self.heatmap_status)

        layout.addWidget(QLabel("3D detected-object view (radar-only, no camera/LiDAR/GNSS overlay -- "
                                  "elevation is genuinely reported by this board's 3TX/4RX config):"))
        self.radar_3d_view = RadarPointCloud3DView()
        layout.addWidget(self.radar_3d_view, 2)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("Live timestamp panel (recent frame intervals):"))
        self.interval_plot = pg.PlotWidget()
        self.interval_plot.setLabel("left", "interval (ms)")
        self.interval_plot.setLabel("bottom", "recent frame #")
        self.interval_curve = self.interval_plot.plot(pen=pg.mkPen(width=2))
        layout.addWidget(self.interval_plot, 1)

        layout.addWidget(QLabel("Raw-data inspector (current frame):"))
        self.raw_inspector = QPlainTextEdit()
        self.raw_inspector.setReadOnly(True)
        layout.addWidget(self.raw_inspector, 1)

        return panel

    def _build_transport_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_button)

        step_button = QPushButton("Step")
        step_button.clicked.connect(lambda: self.playback.step(1))
        layout.addWidget(step_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.playback.reset)
        layout.addWidget(reset_button)

        layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for s in SPEED_CHOICES:
            self.speed_combo.addItem(f"{s}x", userData=s)
        self.speed_combo.setCurrentIndex(SPEED_CHOICES.index(1.0))
        self.speed_combo.currentIndexChanged.connect(
            lambda i: self.playback.set_speed(self.speed_combo.itemData(i))
        )
        layout.addWidget(self.speed_combo)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self.frame_slider, 1)

        self.frame_position_label = QLabel("0 / 0")
        layout.addWidget(self.frame_position_label)

        return bar

    def _build_config_panel(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)

        toggle_row = QHBoxLayout()
        self.config_toggle = QToolButton()
        self.config_toggle.setText("Configuration panel (collapsed) -- click to expand")
        self.config_toggle.setCheckable(True)
        self.config_toggle.clicked.connect(self._toggle_config_panel)
        toggle_row.addWidget(self.config_toggle)
        toggle_row.addStretch(1)
        outer.addLayout(toggle_row)

        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll.setVisible(False)
        self.config_scroll.setMaximumHeight(320)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        self.config_tabs = QTabWidget()
        inner_layout.addWidget(self.config_tabs)
        self.config_scroll.setWidget(inner)
        outer.addWidget(self.config_scroll)
        return container

    def _toggle_config_panel(self):
        expanded = self.config_toggle.isChecked()
        self.config_scroll.setVisible(expanded)
        self.config_toggle.setText(
            "Configuration panel (expanded) -- click to collapse" if expanded
            else "Configuration panel (collapsed) -- click to expand"
        )

    # -- config panel population ---------------------------------------- #

    _CONFIG_SECTIONS = {
        "RF": ["start_frequency", "bandwidth", "range_resolution", "max_range"],
        "Frame": ["configured_frame_rate", "frame_periodicity", "num_loops", "chirps_per_burst"],
        "ADC": ["adc_bits", "adc_output_format", "num_adc_samples", "adc_sample_rate"],
        "Chirp": ["idle_time", "ramp_end_time", "freq_slope", "num_tx_antennas", "num_rx_antennas",
                  "num_virtual_antennas", "velocity_resolution", "max_velocity"],
        "Detection": ["cfar_config_raw"],
        "Output Configuration": ["heatmap_output_enabled", "detected_objects_output_enabled",
                                  "radar_num_range_bins", "radar_num_doppler_bins"],
    }

    def _refresh_config_panel(self):
        self.config_tabs.clear()
        report = self.config_report

        if not report.available:
            tab = QPlainTextEdit()
            tab.setReadOnly(True)
            tab.setPlainText(
                "CONFIGURATION UNAVAILABLE\n\n" + "\n".join(report.errors)
            )
            self.config_tabs.addTab(tab, "Configuration")
            return

        for section_name, field_names in self._CONFIG_SECTIONS.items():
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels(["Field", "Value", "Source"])
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            for field_name in field_names:
                cv = report.get(field_name)
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(field_name))
                if cv is None:
                    table.setItem(row, 1, QTableWidgetItem(_NA))
                    table.setItem(row, 2, QTableWidgetItem(""))
                else:
                    label = cv.display()
                    if cv.derived:
                        label += f"  [DERIVED: {cv.formula}]"
                    table.setItem(row, 1, QTableWidgetItem(label))
                    table.setItem(row, 2, QTableWidgetItem(cv.source))
            self.config_tabs.addTab(table, section_name)

        notes_tab = QPlainTextEdit()
        notes_tab.setReadOnly(True)
        notes_text = "KNOWN CONFIGURATION NOTES / CONFLICTS (reported, not silently resolved):\n\n"
        notes_text += "\n\n".join(f"- {n}" for n in report.notes)
        if report.errors:
            notes_text += "\n\nPARSING NOTES:\n" + "\n".join(report.errors)
        notes_tab.setPlainText(notes_text)
        self.config_tabs.addTab(notes_tab, "Notes / Conflicts")

        self.config_tabs.addTab(self._build_comparison_table(), "Config vs Actual")

    def _build_comparison_table(self) -> QTableWidget:
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Quantity", "Configured", "Actual (recorded)"])
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        def add_row(label, configured, actual):
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(label))
            table.setItem(row, 1, QTableWidgetItem(str(configured) if configured is not None else _NA))
            table.setItem(row, 2, QTableWidgetItem(str(actual) if actual is not None else _NA))

        report = self.config_report
        rate_cv = report.get("configured_frame_rate")
        measured_hz = self.data.rate_stats["mean_hz"] if (self.data and self.data.rate_stats) else None
        add_row(
            "Frame rate (Hz)",
            f"{rate_cv.value:.3f}" if rate_cv and rate_cv.value else _NA,
            f"{measured_hz:.3f}" if measured_hz else _NA,
        )

        bins_cv = report.get("radar_num_range_bins")
        actual_bins = None
        if self.data and self.data.n_frames:
            row0 = self.data.row_at(0)
            actual_bins = int(row0["num_range_bins"]) if row0 is not None else None
        add_row("Range bins", bins_cv.value if bins_cv else _NA, actual_bins)

        dop_cv = report.get("radar_num_doppler_bins")
        actual_dop = None
        if self.data and self.data.n_frames:
            row0 = self.data.row_at(0)
            actual_dop = int(row0["num_doppler_bins"]) if row0 is not None else None
        add_row("Doppler bins", dop_cv.value if dop_cv else _NA, actual_dop)

        heatmap_cv = report.get("heatmap_output_enabled")
        actual_heatmap_present = None
        if self.data and self.data.n_frames:
            row0 = self.data.row_at(0)
            actual_heatmap_present = bool(row0.get("heatmap_path")) if row0 is not None else None
        add_row(
            "Heatmap output enabled",
            heatmap_cv.value if heatmap_cv else _NA,
            actual_heatmap_present,
        )

        return table

    # -- status/error banner --------------------------------------------- #

    def _show_status(self, message: str | None):
        self.status_label.setText(message or "")

    # -- transport handlers ------------------------------------------------ #

    def _toggle_play(self):
        if self.playback.playing:
            self.playback.pause()
        else:
            self.playback.play()

    def _on_playing_changed(self, playing: bool):
        self.play_button.setText("Pause" if playing else "Play")

    def _on_slider_moved(self, value: int):
        if value != self.playback.index:
            self.playback.pause()
            self.playback._set_index(value)

    # -- frame rendering ---------------------------------------------------- #

    def _on_frame_changed(self, index: int):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, (self.data.n_frames - 1) if self.data else 0))
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self.frame_position_label.setText(f"{index} / {max(0, (self.data.n_frames - 1) if self.data else 0)}")

        if self.data is None or self.data.n_frames == 0:
            self._render_no_data()
            return

        row = self.data.row_at(index)
        if row is None:
            self._render_no_data()
            return

        self._update_status_section(row, index)
        self._update_targets_and_speed(row, index)
        self._update_drop_diagnostics(row)
        self._update_heatmap(index)
        self._update_points_view(index)
        self._update_interval_history(row)
        self._update_raw_inspector(row, index)

    def _render_no_data(self):
        for lbl in self._status_fields.values():
            lbl.setText(_NA)
        self.target_table.setRowCount(0)
        self.target_note.setText("NO DATA: no radar frames in this session")
        self.radar_speed_label.setText(_NA)
        self.external_speed_label.setText(_NA)
        self.drop_event_label.setText(_NA)
        self.interval_diag_label.setText(_NA)
        self._clear_heatmap_view()
        self.heatmap_status.setText("NO DATA")
        self.radar_3d_view.clear()
        self.raw_inspector.setPlainText("NO DATA")

    def _update_status_section(self, row, index: int):
        f = self._status_fields
        f["Radar"].setText("TI AWR1843BOOST")
        f["Status"].setText("AVAILABLE")
        f["Frame ID"].setText(str(row.get("frame_id", _NA)))
        f["Timestamp (ns)"].setText(str(int(row["timestamp_ns"])))

        if index > 0:
            prev_ts = int(self.data.radar_df.iloc[index - 1]["timestamp_ns"])
            interval_ms = (int(row["timestamp_ns"]) - prev_ts) / 1e6
            f["Frame interval (ms)"].setText(f"{interval_ms:.2f}")
        else:
            f["Frame interval (ms)"].setText(_NA)

        stats = self.data.rate_stats
        if stats:
            f["Measured rate (mean)"].setText(f"{stats['mean_hz']:.3f} Hz")
            f["Measured rate (median)"].setText(f"{stats['median_hz']:.3f} Hz")
        else:
            f["Measured rate (mean)"].setText(_NA)
            f["Measured rate (median)"].setText(_NA)

        configured_rate = self.config_report.get("configured_frame_rate")
        if configured_rate and configured_rate.value:
            f["Configured rate"].setText(f"{configured_rate.value:.3f} Hz [{configured_rate.source}]")
        else:
            f["Configured rate"].setText(_NA)

        if stats and configured_rate and configured_rate.value:
            diff_pct = 100.0 * (stats["mean_hz"] - configured_rate.value) / configured_rate.value
            f["Effective rate note"].setText(
                f"measured differs from configured by {diff_pct:+.1f}% "
                "(measured rate is genuinely computed from consecutive recorded "
                "timestamps, not assumed)"
            )
        else:
            f["Effective rate note"].setText(_NA)

        if self.data.drop_events:
            f["Dropped frames (recorded)"].setText(
                "; ".join(d["description"] for d in self.data.drop_events)
            )
        else:
            f["Dropped frames (recorded)"].setText("Recorded drop event: none available")

    def _update_targets_and_speed(self, row, index: int):
        points, error = self.data.points_at(index)
        self.target_table.setRowCount(0)
        if error:
            self.target_note.setText(error)
            self.radar_speed_label.setText(_NA)
        elif not points:
            self.target_note.setText("NO DATA: no detected-object points recorded for this frame")
            self.radar_speed_label.setText(_NA)
        else:
            self.target_note.setText("")
            self.target_table.setRowCount(len(points))
            velocities = []
            for i, p in enumerate(points):
                self.target_table.setItem(i, 0, QTableWidgetItem(str(i)))
                self.target_table.setItem(i, 1, QTableWidgetItem(f"{p.get('x', float('nan')):.3f}"))
                self.target_table.setItem(i, 2, QTableWidgetItem(f"{p.get('y', float('nan')):.3f}"))
                v = p.get("velocity_mps")
                self.target_table.setItem(i, 3, QTableWidgetItem(f"{v:.3f}" if v is not None else _NA))
                if v is not None and np.isfinite(v):
                    velocities.append(v)
            if velocities:
                arr = np.array(velocities)
                self.radar_speed_label.setText(
                    f"mean |v|={np.mean(np.abs(arr)):.3f} m/s, max |v|={np.max(np.abs(arr)):.3f} m/s "
                    f"(n={len(arr)} detections)"
                )
            else:
                self.radar_speed_label.setText("NO DATA: no velocity field present on detections")

        ext_speed = self.data.external_reference_speed_mps(int(row["timestamp_ns"]))
        self.external_speed_label.setText(
            f"{ext_speed:.3f} m/s (external reference, MTi)" if ext_speed is not None else _NA
        )

    def _update_drop_diagnostics(self, row):
        current_ts = int(row["timestamp_ns"])
        matching = [d for d in self.data.drop_events if d["timestamp_ns"] == current_ts]
        if matching:
            self.drop_event_label.setText(
                "; ".join(d["description"] for d in matching) + " (recorded at this exact timestamp)"
            )
        elif self.data.drop_events:
            self.drop_event_label.setText(
                "Recorded drop event(s) exist elsewhere in this session: "
                + "; ".join(f"{d['description']} @ ts={d['timestamp_ns']}" for d in self.data.drop_events)
            )
        else:
            self.drop_event_label.setText("Recorded drop event: none available")

        stats = self.data.rate_stats
        if stats:
            self.interval_diag_label.setText(
                f"median interval={stats['median_interval_ms']:.2f}ms, "
                f"p95={stats['p95_interval_ms']:.2f}ms "
                f"(diagnostic only -- not a drop count, not inferred as specific frames)"
            )
        else:
            self.interval_diag_label.setText(_NA)

    def _clear_heatmap_view(self):
        """ImageView.clear() blanks the image but -- confirmed live on a real
        display -- leaves the histogram/colorbar widget showing its last
        (now stale) levels, which would misleadingly suggest data is still
        current. Hiding the histogram whenever there is no image keeps the
        panel honest."""
        self.heatmap_view.clear()
        self.heatmap_view.ui.histogram.hide()

    def _set_heatmap_view(self, arr: np.ndarray):
        self.heatmap_view.ui.histogram.show()
        self.heatmap_view.setImage(arr, autoLevels=True)

    def _update_heatmap(self, index: int):
        heatmap, error = self.data.heatmap_at(index)
        if error:
            self._clear_heatmap_view()
            self.heatmap_status.setText(error)
        else:
            self._set_heatmap_view(heatmap.astype(np.float64))
            self.heatmap_status.setText(f"shape={heatmap.shape}, dtype={heatmap.dtype}")

    def _update_points_view(self, index: int):
        points, error = self.data.points_at(index)
        if error or not points:
            self.radar_3d_view.clear()
            return
        xyz = np.array(
            [[p.get("x", np.nan), p.get("y", np.nan), p.get("z", np.nan)] for p in points],
            dtype=np.float32,
        )
        self.radar_3d_view.set_points(xyz)

    def _update_interval_history(self, row):
        ts = int(row["timestamp_ns"])
        if self._last_ts_for_history is not None:
            interval_ms = (ts - self._last_ts_for_history) / 1e6
            self._interval_history.append(interval_ms)
        self._last_ts_for_history = ts
        if self._interval_history:
            self.interval_curve.setData(list(range(len(self._interval_history))), list(self._interval_history))

    def _update_raw_inspector(self, row, index: int):
        lines = [
            f"frame_id: {row.get('frame_id')}",
            f"timestamp_ns: {int(row['timestamp_ns'])}",
            f"points_path: {row.get('points_path') or '(none)'}",
            f"heatmap_path: {row.get('heatmap_path') or '(none)'}",
            "",
        ]
        heatmap, hm_err = self.data.heatmap_at(index)
        stats = _array_stats(heatmap) if heatmap is not None else {"available": False}
        lines.append("heatmap array: " + (str(stats) if stats["available"] else (hm_err or "unavailable")))

        points, pts_err = self.data.points_at(index)
        if points:
            xs = np.array([p.get("x", np.nan) for p in points])
            stats_x = _array_stats(xs)
            lines.append("points.x array: " + str(stats_x))
        else:
            lines.append("points.x array: " + (pts_err or "unavailable"))

        self.raw_inspector.setPlainText("\n".join(lines))

    # -- live hardware mode ------------------------------------------------ #

    def _toggle_live_radar(self):
        if self._live_receiver is not None:
            self._disconnect_live_radar()
        else:
            self._connect_live_radar()

    def _connect_live_radar(self):
        try:
            RadarReceiver = _import_radar_receiver()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Live radar unavailable",
                                  f"Could not import the radar driver:\n{e}")
            self.live_status_label.setText(f"ERROR: could not import radar driver: {e}")
            return

        self.playback.pause()
        self._last_live_frame_id = None
        self._last_live_ts_ns = None
        self._live_interval_history.clear()
        self.live_status_label.setText("Connecting (sending sensor config, up to a few seconds)...")
        self.live_button.setEnabled(False)

        self._live_receiver = RadarReceiver()
        self._live_thread = QThread(self)
        self._live_worker = LiveRadarWorker(self._live_receiver)
        self._live_worker.moveToThread(self._live_thread)
        self._live_thread.started.connect(self._live_worker.run)
        self._live_worker.started.connect(self._on_live_started)
        self._live_worker.errorOccurred.connect(self._on_live_error)
        self._live_thread.start()

    def _on_live_started(self):
        self.live_button.setEnabled(True)
        self.live_button.setText("Disconnect Live Radar")
        self.live_status_label.setText("Connected. Waiting for first frame...")
        self._live_poll_timer.start()

    def _on_live_error(self, message: str):
        self.live_button.setEnabled(True)
        self.live_button.setText("Connect Live Radar (3TX/4RX)")
        self.live_status_label.setText(f"ERROR: {message}")
        QMessageBox.warning(self, "Live radar error", message)
        self._teardown_live_thread()
        self._live_receiver = None

    def _disconnect_live_radar(self):
        self._live_poll_timer.stop()
        if self._live_receiver is not None:
            try:
                self._live_receiver.stop()
            except Exception:
                pass
        self._teardown_live_thread()
        self._live_receiver = None
        self.live_button.setText("Connect Live Radar (3TX/4RX)")
        self.live_status_label.setText("Not connected.")
        self.live_elevation_label.setText(_NA)

    def _teardown_live_thread(self):
        if self._live_thread is not None:
            self._live_thread.quit()
            self._live_thread.wait(2000)
        self._live_thread = None
        self._live_worker = None

    def _poll_live_frame(self):
        receiver = self._live_receiver
        if receiver is None:
            return
        frame = receiver.get_frame()
        if frame is None:
            self.live_status_label.setText(
                f"Connected -- waiting for first frame... "
                f"(port streaming: {receiver.streaming})"
            )
            return
        if frame.frame_id == self._last_live_frame_id:
            return  # no new frame since last poll
        self._last_live_frame_id = frame.frame_id
        self._render_live_frame(frame, receiver)

    def _render_live_frame(self, frame, receiver):
        ts = int(frame.timestamp_ns)
        interval_ms = None
        if self._last_live_ts_ns is not None:
            interval_ms = (ts - self._last_live_ts_ns) / 1e6
            self._live_interval_history.append(interval_ms)
        self._last_live_ts_ns = ts

        configured_rate = self.config_report.get("configured_frame_rate")
        status_lines = [
            f"LIVE -- streaming: {receiver.streaming}, connected: {receiver.connected}",
            f"frame_id={frame.frame_id}  num_objects={frame.num_objects}",
            f"measured rate (receiver, running average): {receiver.rate:.3f} Hz",
            f"configured rate: {configured_rate.display() if configured_rate else _NA}",
            f"frame interval: {interval_ms:.2f} ms" if interval_ms is not None else "frame interval: N/A / Not available",
            f"dropped frames (live, firmware frame-number gaps): {receiver.dropped_frames}",
        ]
        self.live_status_label.setText("\n".join(status_lines))

        f = self._status_fields
        f["Radar"].setText("TI AWR1843BOOST (LIVE)")
        f["Status"].setText("AVAILABLE (live)")
        f["Frame ID"].setText(str(frame.frame_id))
        f["Timestamp (ns)"].setText(str(ts))
        f["Frame interval (ms)"].setText(f"{interval_ms:.2f}" if interval_ms is not None else _NA)
        f["Measured rate (mean)"].setText(f"{receiver.rate:.3f} Hz (live running average)")
        if self._live_interval_history:
            recent = np.array(self._live_interval_history)
            f["Measured rate (median)"].setText(f"{1000.0 / np.median(recent):.3f} Hz (recent window)")
        else:
            f["Measured rate (median)"].setText(_NA)
        f["Configured rate"].setText(configured_rate.display() + f" [{configured_rate.source}]" if configured_rate else _NA)
        f["Effective rate note"].setText(
            "measured live from the sensor's own frames, not assumed"
        )
        f["Dropped frames (recorded)"].setText(
            f"live firmware frame-number gap count: {receiver.dropped_frames} "
            "(NOT the recorded-session RADAR_DROPPED_FRAMES event -- this is live-only)"
        )

        points = frame.points or []
        self.target_table.setRowCount(len(points))
        velocities = []
        z_values = []
        for i, p in enumerate(points):
            self.target_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.target_table.setItem(i, 1, QTableWidgetItem(f"{p.x:.3f}"))
            self.target_table.setItem(i, 2, QTableWidgetItem(f"{p.y:.3f}"))
            self.target_table.setItem(i, 3, QTableWidgetItem(f"{p.velocity_mps:.3f}"))
            if np.isfinite(p.velocity_mps):
                velocities.append(p.velocity_mps)
            if np.isfinite(p.z):
                z_values.append(p.z)
        self.target_note.setText("" if points else "NO DATA: no detections in latest live frame")
        if velocities:
            arr = np.array(velocities)
            self.radar_speed_label.setText(
                f"mean |v|={np.mean(np.abs(arr)):.3f} m/s, max |v|={np.max(np.abs(arr)):.3f} m/s "
                f"(n={len(arr)} live detections)"
            )
        else:
            self.radar_speed_label.setText("NO DATA: no velocity field on live detections")
        self.external_speed_label.setText(
            "N/A / Not available (no live external-reference sensor connected in this dashboard)"
        )

        if z_values:
            z_arr = np.array(z_values)
            self.live_elevation_label.setText(
                f"z: mean={z_arr.mean():.4f} m, std={z_arr.std():.4f} m, "
                f"min={z_arr.min():.4f} m, max={z_arr.max():.4f} m (n={len(z_arr)})"
            )
        else:
            self.live_elevation_label.setText("NO DATA: no detections to compute elevation from")

        self.drop_event_label.setText(
            "Live mode has no recorded-event metadata (nothing has been recorded yet). "
            f"Live firmware-header drop count so far: {receiver.dropped_frames}"
        )
        stats_text = _NA
        if self._live_interval_history:
            recent = np.array(self._live_interval_history)
            stats_text = (f"median interval={np.median(recent):.2f}ms over last "
                           f"{len(recent)} live frames (diagnostic only)")
        self.interval_diag_label.setText(stats_text)

        rd = frame.range_doppler
        if rd is None:
            self._clear_heatmap_view()
            self.heatmap_status.setText(
                "OPTIONAL DATA UNAVAILABLE: heatmap output is disabled in the current "
                "sensor config (guiMonitor rangeDopplerHeatmap=0)"
            )
        else:
            self._set_heatmap_view(np.asarray(rd, dtype=np.float64))
            self.heatmap_status.setText(f"shape={rd.shape}, dtype={rd.dtype} (live)")

        if points:
            xyz = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
            self.radar_3d_view.set_points(xyz, note="LIVE")
        else:
            self.radar_3d_view.clear()

        if self._live_interval_history:
            hist = list(self._live_interval_history)
            self.interval_curve.setData(list(range(len(hist))), hist)

        lines = [
            f"[LIVE] frame_id: {frame.frame_id}",
            f"timestamp_ns: {ts}",
            f"num_objects: {frame.num_objects}",
            "",
        ]
        rd_stats = _array_stats(rd) if rd is not None else {"available": False}
        lines.append("range_doppler array: " + (str(rd_stats) if rd_stats["available"]
                                                  else "unavailable (heatmap disabled in current config)"))
        if points:
            xs_arr = np.array([p.x for p in points])
            lines.append("points.x array: " + str(_array_stats(xs_arr)))
            zs_arr = np.array([p.z for p in points])
            lines.append("points.z array: " + str(_array_stats(zs_arr)))
        else:
            lines.append("points.x array: unavailable (no detections)")
        self.raw_inspector.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self._disconnect_live_radar()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv)

    dataset_root = Path(argv[0]) if len(argv) >= 1 else DEFAULT_DATASET_ROOT
    session_dir = Path(argv[1]) if len(argv) >= 2 else None

    window = RadarDashboard(dataset_root=dataset_root, session_dir=session_dir)
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
