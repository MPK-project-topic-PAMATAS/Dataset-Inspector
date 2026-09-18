"""Interactive 3D point-cloud viewer for recorded/received radar detections.

Shared by both the main Dataset Inspector (main_window.py) and the
standalone radar diagnostic dashboard (tests/radar_dashboard.py) -- extracted
here specifically so neither duplicates the rendering logic.

Pure presentation widget, exactly like lidar_3d_view.Lidar3DView: it never
reads a radar file, never talks to RecordingSession/SyncIndex, and never
performs a timestamp lookup of its own. The caller is the sole authority on
*which* recorded/received sample is current; this widget only renders the
(x, y, z) array it's given.

The AWR1843BOOST, in whatever configuration produced the data passed in, may
or may not report a meaningful third axis -- this widget never fabricates
one: if every z happens to be 0.0 (as in a 2TX recording) the cloud is simply
flat; if z genuinely varies (as confirmed live with this project's 3TX/4RX
config), it's rendered as received. No signal processing of any kind happens
here.

Rendering library: pyqtgraph.opengl (GLViewWidget/GLScatterPlotItem), the
same one lidar_3d_view.py already uses -- no new dependency.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RadarPointCloud3DView(QWidget):
    """Created once by the caller and reused for every frame -- see
    set_points()/clear(). The GL widget, its scatter item, and its axis
    gizmo are never destroyed/recreated; only the point buffer and header
    text change. Radar frames are a handful of points, so unlike the LiDAR
    viewer this needs no LOD/performance handling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.header_label = QLabel("Radar 3D detections: no data")
        outer.addWidget(self.header_label)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setMinimumHeight(260)
        self.gl_view.setCameraPosition(distance=10, elevation=25, azimuth=-60)
        outer.addWidget(self.gl_view, stretch=1)

        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=np.float32))
        self.gl_view.addItem(self.scatter)

        self.axis_item = gl.GLAxisItem()
        self.axis_item.setSize(2, 2, 2)
        self.gl_view.addItem(self.axis_item)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Color: elevation (z)"))
        controls.addStretch(1)
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        controls.addWidget(self.reset_button)
        self.fit_button = QPushButton("Fit to Points")
        self.fit_button.clicked.connect(self.fit_to_points)
        controls.addWidget(self.fit_button)
        outer.addLayout(controls)

        self._current_points = np.zeros((0, 3), dtype=np.float32)

    def set_points(self, points_xyz, note: str = ""):
        """`points_xyz` is an (N,3) array of already-recorded/received
        detections (radar-only -- never fused with any other sensor).
        Never mutates the array passed in."""
        pts = np.asarray(points_xyz, dtype=np.float32) if points_xyz is not None and len(points_xyz) else np.zeros((0, 3), dtype=np.float32)
        if len(pts):
            finite = np.isfinite(pts).all(axis=1)
            clean = pts[finite]
            excluded = len(pts) - len(clean)
        else:
            clean = pts
            excluded = 0
        self._current_points = clean

        if len(clean) == 0:
            self.scatter.setData(pos=clean)
        else:
            self.scatter.setData(pos=clean, size=10, color=self._colors_for(clean))

        text = f"Radar 3D detections: {len(clean)} point(s)"
        if excluded:
            text += f" ({excluded} non-finite excluded from rendering)"
        if note:
            text += f"  -- {note}"
        self.header_label.setText(text)

    def clear(self):
        self._current_points = np.zeros((0, 3), dtype=np.float32)
        self.scatter.setData(pos=self._current_points)
        self.header_label.setText("Radar 3D detections: no data")

    def reset_view(self):
        self.gl_view.setCameraPosition(distance=10, elevation=25, azimuth=-60)
        self.gl_view.opts["center"] = QVector3D(0, 0, 0)
        self.gl_view.update()

    def fit_to_points(self):
        pts = self._current_points
        if len(pts) == 0:
            return
        centroid = pts.mean(axis=0)
        center = QVector3D(float(centroid[0]), float(centroid[1]), float(centroid[2]))
        if len(pts) == 1:
            distance = 2.0
        else:
            extent = max(float(np.max(pts.max(axis=0) - pts.min(axis=0))), 1e-3)
            distance = max(extent * 2.0, 1.0)
        self.gl_view.opts["distance"] = distance
        self.gl_view.opts["center"] = center
        self.gl_view.update()

    def _colors_for(self, pts: np.ndarray) -> np.ndarray:
        z = pts[:, 2]
        span = max(float(z.max() - z.min()), 1e-6)
        z_norm = np.clip((z - z.min()) / span, 0.0, 1.0)
        colors = np.empty((len(pts), 4), dtype=np.float32)
        colors[:, 0] = z_norm
        colors[:, 1] = 0.47 + 0.39 * (1.0 - z_norm)
        colors[:, 2] = 1.0 - z_norm
        colors[:, 3] = 0.9
        return colors
