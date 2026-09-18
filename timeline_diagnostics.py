"""Read-only Timeline Diagnostics widget.

Visualizes, against the camera master timeline, each available sensor's
recorded temporal coverage, its currently SyncIndex-selected sample, the
synchronization delta SensorHealthModel already computed, and recorded
drop events -- all from data the application already loaded. It never
reads a file, never decodes anything, never searches for a nearest sample,
and never invokes DatasetIntegrityValidator or rebuilds SyncIndex.

Dependency direction (never the reverse): Session -> SyncIndex ->
SensorHealth -> this widget. Everything this widget draws is handed to it
as plain data (`set_session`/`set_current`) by MainWindow; it has no
reference to RecordingSession, SyncIndex, or SvoReader itself.

Event markers are rendered by embedding the existing EventMarkerStrip
rather than reimplementing marker drawing/click handling a second time.
Hovering an event reads that strip's already-set marker list (read-only
introspection of data it already holds) rather than adding a second event
data path.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from event_marker_strip import EventMarkerStrip
from sensor_health import SensorState

_STATE_COLORS = {
    SensorState.AVAILABLE: QColor(90, 200, 90),
    SensorState.STALE: QColor(230, 160, 40),
    SensorState.DROPPED: QColor(220, 60, 60),
    SensorState.INVALID: QColor(220, 60, 60),
    SensorState.NO_DATA: QColor(110, 110, 110),
    SensorState.NO_FIX: QColor(150, 150, 150),
    SensorState.UNAVAILABLE: QColor(110, 110, 110),
}
_DEFAULT_STATE_COLOR = QColor(200, 200, 200)

_LANE_ORDER = ("Camera", "Radar", "Lidar", "MTi-8", "ZED IMU", "GNSS")
_EVENT_STRIP_HEIGHT = 12
_LANE_HEIGHT = 22
_AXIS_HEIGHT = 14
_TOP_MARGIN = 4
_LABEL_WIDTH = 70
_STATUS_WIDTH = 150
_HOVER_FRACTION_TOLERANCE = 0.03


def _seconds_str(ns: int) -> str:
    """Plain unit conversion for display only (ns -> seconds); never fed
    back into any timestamp, sample selection, or synchronization logic."""
    return f"{ns / 1e9:.3f}s"


class TimelineDiagnostics(QWidget):
    """A compact, session-wide diagnostic overview -- not a raw-data plot.

    Draws one coverage bar per available sensor, a shared elapsed-time
    axis, a single vertical cursor at the current camera timestamp shared
    by every lane, a small marker plus a text status/delta readout on each
    sensor's lane at its currently SyncIndex-selected sample, and the
    existing EventMarkerStrip along the bottom for recorded drop events.
    """

    seekRequested = Signal(object)  # timestamp_ns; forwarded to MainWindow's existing _seek_to_timestamp
    detailsRequested = Signal()     # widget clicked outside a marker -> ask for the expanded view

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(_AXIS_HEIGHT + _TOP_MARGIN + _LANE_HEIGHT * len(_LANE_ORDER) + _EVENT_STRIP_HEIGHT + 4)
        self.setToolTip("Timeline diagnostics: sensor coverage, current sync position, recorded events")
        self.setMouseTracking(True)

        self._camera_first_ns: int | None = None
        self._camera_last_ns: int | None = None
        self._coverage: dict[str, tuple[int, int]] = {}
        self._snapshot = None  # SensorHealthSnapshot | None

        # Layout computed each paint, reused for hover hit-testing so lane
        # positions are never duplicated between drawing and interaction.
        self._lane_rects: dict[str, tuple[int, int]] = {}  # lane -> (y_top, y_bottom)
        self._plot_left = _LABEL_WIDTH
        self._plot_width = 1

        # Reused, not reimplemented -- see module docstring.
        self._event_strip = EventMarkerStrip(self)
        self._event_strip.markerClicked.connect(self.seekRequested)
        self._event_strip.setMouseTracking(True)
        self._event_strip.installEventFilter(self)

    # ------------------------------------------------------------ inputs ---
    def set_session(self, camera_first_ns: int, camera_last_ns: int,
                     coverage: dict[str, tuple[int, int]], drop_events) -> None:
        """Replaces all session-scoped state atomically -- called once per
        commit, from already-loaded arrays (frame/radar/lidar/mti/svo
        timestamps) and SensorHealthModel.drop_events. Never touches disk.
        """
        self._camera_first_ns = camera_first_ns
        self._camera_last_ns = camera_last_ns
        self._coverage = dict(coverage)
        self._snapshot = None

        span = self._span()
        markers = []
        for ev in drop_events:
            frac = (ev.timestamp_ns - camera_first_ns) / span
            color = QColor(255, 140, 0) if "RADAR" in ev.event_type else QColor(0, 200, 255)
            markers.append((frac, color, ev.timestamp_ns, f"{ev.event_type}: {ev.description}"))
        self._event_strip.set_markers(markers)
        self._layout_event_strip()
        self.update()

    def set_current(self, snapshot) -> None:
        """Called on every seek with the current SensorHealthSnapshot -- an
        O(1) array read already done by SensorHealthModel, not recomputed
        here. No disk I/O, no SyncIndex rebuild, no worker request."""
        self._snapshot = snapshot
        self.update()

    def clear(self) -> None:
        self._camera_first_ns = None
        self._camera_last_ns = None
        self._coverage = {}
        self._snapshot = None
        self._lane_rects = {}
        self._event_strip.set_markers([])
        self.update()

    # ------------------------------------------------------------ layout ---
    def resizeEvent(self, event) -> None:
        self._layout_event_strip()
        super().resizeEvent(event)

    def _layout_event_strip(self) -> None:
        self._event_strip.setGeometry(
            _LABEL_WIDTH, self.height() - _EVENT_STRIP_HEIGHT, max(self.width() - _LABEL_WIDTH, 1), _EVENT_STRIP_HEIGHT,
        )

    def _span(self) -> int:
        """Camera range in ns, guarded against zero/negative so a
        degenerate or single-timestamp recording never divides by zero."""
        if self._camera_first_ns is None or self._camera_last_ns is None:
            return 1
        return max(self._camera_last_ns - self._camera_first_ns, 1)

    def _fraction(self, timestamp_ns) -> float | None:
        if self._camera_first_ns is None or timestamp_ns is None:
            return None
        return (timestamp_ns - self._camera_first_ns) / self._span()

    def _clamped_x(self, plot_left: int, plot_width: int, frac: float) -> int:
        return plot_left + int(max(0.0, min(1.0, frac)) * plot_width)

    def _diag_for_lane(self, lane: str):
        if self._snapshot is None:
            return None
        return {
            "Camera": self._snapshot.camera,
            "Radar": self._snapshot.radar,
            "Lidar": self._snapshot.lidar,
            "MTi-8": self._snapshot.mti,
            "ZED IMU": self._snapshot.zed_imu,
            "GNSS": self._snapshot.gnss,
        }.get(lane)

    @staticmethod
    def _format_delta(diag) -> str:
        if diag is None or diag.delta_ms is None:
            return ""
        sign = "+" if diag.delta_ms >= 0 else ""
        return f"{sign}{diag.delta_ms:.1f} ms"

    # ------------------------------------------------------------ paint  ---
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        width = max(self.width(), 1)
        plot_left = _LABEL_WIDTH
        plot_width = max(width - plot_left - _STATUS_WIDTH - 8, 1)
        self._plot_left = plot_left
        self._plot_width = plot_width

        self._paint_axis(painter, plot_left, plot_width)

        self._lane_rects = {}
        y = _AXIS_HEIGHT + _TOP_MARGIN
        for lane in _LANE_ORDER:
            if lane != "Camera" and lane not in self._coverage:
                continue  # no fake lanes for sensors with no recorded data
            self._lane_rects[lane] = (y, y + _LANE_HEIGHT)
            self._paint_lane(painter, lane, y, plot_left, plot_width)
            y += _LANE_HEIGHT

        # One cursor, shared by every lane, at the current camera timestamp.
        diag_camera = self._diag_for_lane("Camera")
        if diag_camera is not None and diag_camera.sensor_timestamp_ns is not None:
            frac = self._fraction(diag_camera.sensor_timestamp_ns)
            if frac is not None:
                x = self._clamped_x(plot_left, plot_width, frac)
                pen = QPen(QColor(255, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(x, _AXIS_HEIGHT, x, y)
                self._paint_cursor_label(painter, x, plot_left, plot_width, diag_camera.sensor_timestamp_ns)

    def _paint_axis(self, painter: QPainter, plot_left: int, plot_width: int) -> None:
        """Compact elapsed-time labels (seconds since camera start) at the
        two ends of the shared axis -- display only; the underlying
        nanosecond timestamps this is derived from are never altered."""
        if self._camera_first_ns is None:
            return
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(plot_left, _AXIS_HEIGHT - 2, "0.000s")
        end_label = _seconds_str(self._camera_last_ns - self._camera_first_ns)
        metrics = QFontMetrics(painter.font())
        end_x = plot_left + plot_width - metrics.horizontalAdvance(end_label)
        painter.drawText(max(end_x, plot_left), _AXIS_HEIGHT - 2, end_label)

    def _paint_cursor_label(self, painter: QPainter, x: int, plot_left: int, plot_width: int, camera_ts: int) -> None:
        if self._camera_first_ns is None:
            return
        label = _seconds_str(camera_ts - self._camera_first_ns)
        metrics = QFontMetrics(painter.font())
        label_x = min(max(x - metrics.horizontalAdvance(label) // 2, plot_left), plot_left + plot_width - metrics.horizontalAdvance(label))
        painter.setPen(QColor(255, 255, 0))
        painter.drawText(label_x, _AXIS_HEIGHT - 2, label)

    def _paint_lane(self, painter: QPainter, lane: str, y: int, plot_left: int, plot_width: int) -> None:
        mid = y + _LANE_HEIGHT // 2
        painter.setPen(QColor(210, 210, 210))
        painter.drawText(4, mid + 4, lane)

        rng = self._coverage.get(lane)
        if rng is not None:
            first_ns, last_ns = rng
            f0 = self._fraction(first_ns)
            f1 = self._fraction(last_ns)
            if f0 is not None and f1 is not None:
                x0 = self._clamped_x(plot_left, plot_width, f0)
                x1 = self._clamped_x(plot_left, plot_width, f1)
                pen = QPen(QColor(90, 140, 200))
                pen.setWidth(4)
                painter.setPen(pen)
                # A one-sample sensor (or a sub-pixel-width span) still
                # draws a visible dash rather than disappearing entirely.
                painter.drawLine(min(x0, x1), mid, max(x0, x1) + (1 if x0 == x1 else 0), mid)

        diag = self._diag_for_lane(lane)
        color = _STATE_COLORS.get(diag.state, _DEFAULT_STATE_COLOR) if diag is not None else _DEFAULT_STATE_COLOR

        if diag is not None and diag.sensor_timestamp_ns is not None and lane != "Camera":
            frac = self._fraction(diag.sensor_timestamp_ns)
            if frac is not None:
                x = self._clamped_x(plot_left, plot_width, frac)
                painter.setPen(QPen(color, 1))
                painter.setBrush(color)
                painter.drawEllipse(x - 3, mid - 3, 6, 6)

        if diag is not None and lane != "Camera":
            status_x = plot_left + plot_width + 8
            delta_text = self._format_delta(diag)
            text = f"{delta_text}  {diag.state.display()}" if delta_text else diag.state.display()
            painter.setPen(color)
            metrics = QFontMetrics(painter.font())
            painter.drawText(status_x, mid + 4, metrics.elidedText(text, Qt.ElideRight, _STATUS_WIDTH - 4))

    # ------------------------------------------------------------ input  ---
    def mousePressEvent(self, event) -> None:
        # The embedded EventMarkerStrip handles its own clicks (it occupies
        # only the bottom strip); anywhere else in this widget requests the
        # expanded diagnostics view instead of seeking or altering anything.
        self.detailsRequested.emit()

    def mouseMoveEvent(self, event) -> None:
        lane = self._lane_at_y(self._event_y(event))
        if lane is None:
            QToolTip.hideText()
            super().mouseMoveEvent(event)
            return
        diag = self._diag_for_lane(lane)
        if diag is None:
            QToolTip.hideText()
            super().mouseMoveEvent(event)
            return
        QToolTip.showText(self._event_global_pos(event), self._tooltip_for(lane, diag), self)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._event_strip and event.type() == QEvent.Type.MouseMove:
            self._handle_event_strip_hover(event)
        elif obj is self._event_strip and event.type() == QEvent.Type.Leave:
            QToolTip.hideText()
        return super().eventFilter(obj, event)

    def _handle_event_strip_hover(self, event) -> None:
        # Read-only introspection of markers the strip already holds (set
        # via its own public set_markers()) -- no second event data path.
        markers = getattr(self._event_strip, "_markers", [])
        if not markers:
            QToolTip.hideText()
            return
        width = max(self._event_strip.width() - 1, 1)
        frac = self._event_x(event) / width
        nearest = min(markers, key=lambda m: abs(m[0] - frac))
        if abs(nearest[0] - frac) > _HOVER_FRACTION_TOLERANCE:
            QToolTip.hideText()
            return
        _frac, _color, timestamp_ns, label = nearest
        text = f"{label}\ntimestamp: {timestamp_ns / 1e9:.6f} s"
        QToolTip.showText(self._event_global_pos(event), text, self._event_strip)

    def _tooltip_for(self, lane: str, diag) -> str:
        lines = [lane, f"Status: {diag.state.display()}"]
        if diag.sample_index is not None:
            lines.append(f"Sample: {diag.sample_index}")
        if diag.sensor_timestamp_ns is not None:
            lines.append(f"Sensor timestamp: {diag.sensor_timestamp_ns / 1e9:.6f} s")
        delta_text = self._format_delta(diag)
        if delta_text:
            lines.append(f"Δt: {delta_text}" + (f"  ({diag.sync_class})" if diag.sync_class else ""))
        if diag.detail:
            lines.append(diag.detail)
        return "\n".join(lines)

    def _lane_at_y(self, y: int) -> str | None:
        for lane, (top, bottom) in self._lane_rects.items():
            if top <= y < bottom:
                return lane
        return None

    @staticmethod
    def _event_y(event) -> int:
        pos = event.position() if hasattr(event, "position") else event.pos()
        return int(pos.y())

    @staticmethod
    def _event_x(event) -> float:
        pos = event.position() if hasattr(event, "position") else event.pos()
        return float(pos.x())

    @staticmethod
    def _event_global_pos(event):
        if hasattr(event, "globalPosition"):
            return event.globalPosition().toPoint()
        return event.globalPos()

    # ------------------------------------------------------------ reads  ---
    def coverage(self) -> dict[str, tuple[int, int]]:
        """Read-only snapshot of the current session's per-lane coverage
        ranges (first/last recorded timestamp, ns), for callers building an
        expanded details view without reaching into private state."""
        return dict(self._coverage)

    def current_state_summary(self) -> str:
        """Plain-text one-liner for a status bar / tooltip; not used for
        any decision, purely a display convenience."""
        if self._snapshot is None:
            return "no session"
        parts = []
        for lane in _LANE_ORDER:
            diag = self._diag_for_lane(lane)
            if diag is not None:
                parts.append(f"{lane}: {diag.state.display()}")
        return "  ".join(parts)
