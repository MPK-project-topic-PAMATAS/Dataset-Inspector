"""A thin, read-only marker strip drawn alongside the existing playback
scrubber to show where recorded events (e.g. RADAR_DROPPED_FRAMES) fall on
the camera timeline. It never replaces the scrubber and never affects
playback by itself -- clicking a marker only requests a seek through the
same seek path the existing event list already uses.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal


class EventMarkerStrip(QWidget):
    # timestamp_ns of the marker nearest the click. `object`, not `int`: a
    # recorded timestamp_ns (~1.8e18) overflows Qt's 32-bit `int` signal type.
    markerClicked = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(12)
        self.setToolTip("Recorded events on the camera timeline (click to seek)")
        self._markers: list = []  # (fraction 0..1, QColor, timestamp_ns, label)

    def set_markers(self, markers) -> None:
        self._markers = list(markers)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w = max(self.width() - 1, 1)
        h = self.height()
        for frac, color, _ts, _label in self._markers:
            x = int(max(0.0, min(1.0, frac)) * w)
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(x, 0, x, h)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if not self._markers:
            return
        w = max(self.width() - 1, 1)
        clicked_frac = event.position().x() / w
        _frac, _color, ts, _label = min(self._markers, key=lambda m: abs(m[0] - clicked_frac))
        self.markerClicked.emit(ts)
