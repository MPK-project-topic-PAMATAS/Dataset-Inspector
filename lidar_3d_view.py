"""Interactive 3D point-cloud viewer for the recorded Ouster LiDAR scan
selected for the current camera frame.

This is a pure presentation widget, exactly like sensor_health.py and
timeline_diagnostics.py: it never reads a PCAP/JSON file, never talks to
LidarReader/RecordingSession/SyncIndex, and never performs a timestamp
lookup of its own. MainWindow continues to be the only place that decides
*which* recorded scan is current (via the existing SyncIndex mapping); this
widget only receives that already-selected point array plus already-
computed diagnostic metadata (a sensor_health.SensorDiagnostic) and renders
it.

Coordinate frame: rendered exactly as LidarReader/XYZLut already produces
it -- the Ouster sensor's own frame (X/Y/Z per the sensor's documented
`lidar_to_sensor_transform`, i.e. Z already conventionally "up" for this
sensor family). Nothing here transforms, registers, fuses, or relabels
those coordinates; the axis gizmo is generic X/Y/Z, not a vehicle-relative
label, since this module has no basis for claiming "front"/"left"/"up"
beyond what the sensor's own frame already is.

Rendering library: pyqtgraph.opengl (GLViewWidget/GLScatterPlotItem), the
existing project's own plotting library extended with its bundled OpenGL
module. The only new dependency is PyOpenGL, which pyqtgraph.opengl requires
to draw anything at all -- there is no heavier alternative already present
in this project, and PyOpenGL is the minimal, standard way to satisfy that
requirement.
"""
from __future__ import annotations

import logging

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sensor_health import SensorDiagnostic

logger = logging.getLogger(__name__)

_DEFAULT_POINT_SIZE = 2.0
_DEFAULT_AZIMUTH = -60.0
_DEFAULT_ELEVATION = 25.0
_MIN_FIT_DISTANCE = 1.0
_MIN_FIT_EXTENT = 1e-3


class _PointCloudGLView(gl.GLViewWidget):
    """GLViewWidget's own defaults already give LEFT-drag orbit and MOUSE
    WHEEL zoom; this subclass only adds the two interactions it doesn't
    provide out of the box: RIGHT-drag pan (it only maps MIDDLE-drag), and
    double-click / 'R' as a reset shortcut. No dataset access of any kind
    happens in any of these handlers -- they only change the view transform.
    """

    resetRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseMoveEvent(self, ev):
        if ev.buttons() == Qt.MouseButton.RightButton:
            lpos = ev.position() if hasattr(ev, "position") else ev.localPos()
            if not hasattr(self, "mousePos"):
                self.mousePos = lpos
            diff = lpos - self.mousePos
            self.mousePos = lpos
            self.pan(diff.x(), diff.y(), 0, relative="view-upright")
        else:
            super().mouseMoveEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self.resetRequested.emit()
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev):
        if ev.text().lower() == "r":
            self.resetRequested.emit()
        else:
            super().keyPressEvent(ev)


class Lidar3DView(QWidget):
    """Created once by MainWindow and reused for every frame -- see
    set_points()/show_missing(). The GL widget, its scatter item, and its
    axis gizmo are never destroyed/recreated on a per-frame basis; only the
    point buffer and the header text change.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.header_label = QLabel("LiDAR: no session loaded")
        self.header_label.setStyleSheet("color: #ccc;")
        outer.addWidget(self.header_label)

        self.gl_view = _PointCloudGLView()
        self.gl_view.setMinimumHeight(260)
        self.gl_view.opts["distance"] = 10
        self.gl_view.setCameraPosition(distance=10, elevation=_DEFAULT_ELEVATION, azimuth=_DEFAULT_AZIMUTH)
        self.gl_view.resetRequested.connect(self.reset_view)
        outer.addWidget(self.gl_view, stretch=1)

        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=np.float32))
        self.gl_view.addItem(self.scatter)

        # A generic X/Y/Z gizmo -- see module docstring for why these are
        # not labeled with vehicle-relative directions.
        self.axis_item = gl.GLAxisItem()
        self.axis_item.setSize(1, 1, 1)
        self.gl_view.addItem(self.axis_item)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Point size:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(int(_DEFAULT_POINT_SIZE))
        self.point_size_slider.setMaximumWidth(120)
        self.point_size_slider.valueChanged.connect(self._on_point_size_changed)
        controls.addWidget(self.point_size_slider)

        controls.addWidget(QLabel("Color:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Height", "Uniform"])
        self.color_mode_combo.currentTextChanged.connect(self._on_color_mode_changed)
        controls.addWidget(self.color_mode_combo)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        controls.addWidget(self.reset_view_btn)

        self.fit_btn = QPushButton("Fit to Cloud")
        self.fit_btn.clicked.connect(self.fit_to_cloud)
        controls.addWidget(self.fit_btn)
        controls.addStretch(1)
        outer.addLayout(controls)

        self._point_size = float(_DEFAULT_POINT_SIZE)
        self._color_mode = "Height"
        self._current_points = np.zeros((0, 3), dtype=np.float32)
        self._has_shown_any_cloud = False  # gates the one-time initial auto-fit (see set_points)
        self._default_center = QVector3D(0, 0, 0)
        self._default_distance = 10.0

    # -------------------------------------------------------------- inputs ---
    def set_points(self, points, scan_index=None, timestamp_ns=None, diagnostic: SensorDiagnostic | None = None):
        """Replaces the displayed point cloud with `points` (an (N,3) array
        already selected by MainWindow via the existing SyncIndex path).

        `diagnostic` is the existing sensor_health.SensorDiagnostic for
        LiDAR at the current frame, if available -- its delta/state are
        shown verbatim; nothing here recomputes a synchronization value.

        Never touches disk, never blocks, never recreates the GL widget.
        """
        if points is None:
            self.show_missing("No LiDAR data for this frame")
            return

        pts = np.asarray(points, dtype=np.float32)
        total = len(pts)
        if total:
            # Rendering guard only, not dataset cleaning: a GL vertex buffer
            # cannot contain NaN/Inf without producing undefined/garbage
            # geometry. This exclusion is presentation-only -- it never
            # writes anywhere, and the array MainWindow was given (and
            # whatever it does with it elsewhere) is untouched.
            finite = np.isfinite(pts).all(axis=1)
            clean = pts[finite]
        else:
            clean = pts
        excluded = total - len(clean)

        self._current_points = clean
        self._apply_rendering()

        self._update_header(scan_index, timestamp_ns, diagnostic, displayed=len(clean), excluded=excluded)

        if not self._has_shown_any_cloud and len(clean) > 0:
            # One sensible initial framing per session (see class docstring
            # and requirement 13) -- never repeated on later frames, so the
            # user's own orientation/zoom while scrubbing is never disturbed.
            self.fit_to_cloud()
            self._has_shown_any_cloud = True

    def show_missing(self, message: str):
        """No selected LiDAR sample for this frame/session: shows an empty
        scene with an explicit message rather than fabricating any points
        (an empty (0,3) array is the honest "nothing recorded/selected"
        representation, not a manufactured cloud)."""
        self._current_points = np.zeros((0, 3), dtype=np.float32)
        self.scatter.setData(pos=self._current_points)
        self.header_label.setText(f"LiDAR: {message}")

    def set_session_info(self, text: str):
        """Lightweight header-only update (e.g. 'N scans loaded'), used
        right after a session commits, before any frame has been rendered."""
        self.header_label.setText(f"LiDAR: {text}")

    def clear(self):
        """Called on session replacement so the previous session's cloud
        and one-time auto-fit state are never carried into the new one."""
        self._current_points = np.zeros((0, 3), dtype=np.float32)
        self.scatter.setData(pos=self._current_points)
        self._has_shown_any_cloud = False
        self.header_label.setText("LiDAR: no session loaded")

    # ------------------------------------------------------------ controls ---
    def _on_point_size_changed(self, value: int):
        self._point_size = float(value)
        self._apply_rendering()

    def _on_color_mode_changed(self, mode: str):
        self._color_mode = mode
        self._apply_rendering()

    def reset_view(self):
        """Restores a sensible default orientation. Only ever changes the
        camera transform -- the point buffer is untouched."""
        distance, center = self._fit_parameters()
        self.gl_view.setCameraPosition(distance=distance, elevation=_DEFAULT_ELEVATION, azimuth=_DEFAULT_AZIMUTH)
        self.gl_view.opts["center"] = center
        self.gl_view.update()

    def fit_to_cloud(self):
        """Recenters/rescales the camera to frame the current cloud, keeping
        the current orientation. Only ever changes the camera transform."""
        distance, center = self._fit_parameters()
        self.gl_view.opts["distance"] = distance
        self.gl_view.opts["center"] = center
        self.gl_view.update()

    # -------------------------------------------------------------- helpers ---
    def _fit_parameters(self):
        pts = self._current_points
        if len(pts) == 0:
            return self._default_distance, self._default_center
        centroid = pts.mean(axis=0)
        center = QVector3D(float(centroid[0]), float(centroid[1]), float(centroid[2]))
        if len(pts) == 1:
            return max(_MIN_FIT_DISTANCE, 2.0), center
        extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        extent = max(extent, _MIN_FIT_EXTENT)
        distance = max(extent * 1.5, _MIN_FIT_DISTANCE)
        return distance, center

    def _apply_rendering(self):
        pts = self._current_points
        if len(pts) == 0:
            self.scatter.setData(pos=pts, size=self._point_size)
            return
        color = self._colors_for(pts)
        self.scatter.setData(pos=pts, size=self._point_size, color=color)

    def _colors_for(self, pts: np.ndarray) -> np.ndarray:
        if self._color_mode == "Uniform":
            return np.tile(np.array([[0.8, 0.8, 0.85, 1.0]], dtype=np.float32), (len(pts), 1))

        # Height coloring: presentation-only derived values (a color per
        # point from its own already-recorded Z), never written back or
        # stored -- computed fresh here and discarded once drawn. Same
        # gradient formula the previous 2D top-down view used, so the
        # visual convention stays consistent.
        z = pts[:, 2]
        span = max(float(z.max() - z.min()), 1e-6)
        z_norm = np.clip((z - z.min()) / span, 0.0, 1.0)
        colors = np.empty((len(pts), 4), dtype=np.float32)
        colors[:, 0] = z_norm
        colors[:, 1] = 0.47 + 0.39 * (1.0 - z_norm)
        colors[:, 2] = 1.0 - z_norm
        colors[:, 3] = 0.8
        return colors

    def _update_header(self, scan_index, timestamp_ns, diagnostic, displayed: int, excluded: int):
        parts = ["LiDAR: Ouster OS-0-64"]
        if scan_index is not None:
            parts.append(f"Scan {scan_index}")
        if timestamp_ns is not None:
            parts.append(f"{timestamp_ns / 1e9:.6f} s")
        if diagnostic is not None:
            if diagnostic.delta_ms is not None:
                sign = "+" if diagnostic.delta_ms >= 0 else ""
                parts.append(f"Δt {sign}{diagnostic.delta_ms:.1f} ms")
            parts.append(diagnostic.state.display())
        count_text = f"{displayed:,} points"
        if excluded:
            count_text += f" ({excluded:,} non-finite excluded from rendering)"
        parts.append(count_text)
        self.header_label.setText("   ".join(parts))
