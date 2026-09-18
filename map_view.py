"""An OpenStreetMap (Leaflet) panel showing the GNSS track and current position."""
from __future__ import annotations

import json

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView

_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  html, body, #map { height: 100%; margin: 0; background: #111; }
</style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  var map = L.map('map').setView([0, 0], 2);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  var track = L.polyline([], {color: '#ffcc00', weight: 3}).addTo(map);
  var marker = L.circleMarker([0, 0], {radius: 6, color: '#00ffff', fillColor: '#00ffff', fillOpacity: 1}).addTo(map);
  var fitted = false;

  function setTrack(coords) {
      track.setLatLngs(coords);
      if (coords.length > 0 && !fitted) {
          map.fitBounds(track.getBounds(), {padding: [20, 20]});
          fitted = true;
      }
  }

  function setPosition(lat, lon) {
      marker.setLatLng([lat, lon]);
  }
</script>
</body>
</html>
"""


class MapView(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ready = False
        self._pending_track = None
        self._pending_position = None
        self.loadFinished.connect(self._on_load_finished)
        # A real base URL (not about:blank) is required for the CDN-loaded
        # Leaflet assets to resolve relative to https:// correctly.
        self.setHtml(_HTML, baseUrl=QUrl("https://localhost/"))

    def _on_load_finished(self, ok: bool):
        self._ready = ok
        if not ok:
            return
        if self._pending_track is not None:
            self.set_track(self._pending_track)
            self._pending_track = None
        if self._pending_position is not None:
            self.set_position(*self._pending_position)
            self._pending_position = None

    def set_track(self, latlon_pairs):
        if not self._ready:
            self._pending_track = latlon_pairs
            return
        coords = json.dumps(latlon_pairs)
        self.page().runJavaScript(f"setTrack({coords});")

    def set_position(self, lat: float, lon: float):
        if not self._ready:
            self._pending_position = (lat, lon)
            return
        self.page().runJavaScript(f"setPosition({lat}, {lon});")
