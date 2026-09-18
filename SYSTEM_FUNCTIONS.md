# Dataset Inspector — System Functions

2026-09-18 · @Someone

## Overview

**Dataset Inspector** is a desktop GUI application for reviewing multi-sensor vehicle recordings after the fact. It works like a media player: open a recording session, scrub a timeline with play/pause/rewind/step, and watch every sensor's data update in sync at the current point in time.

Each session under `output_multi-record/output/<session_id>/` bundles recordings from five independent sensors, each logging at its own native rate with no synchronization applied during capture:

- **ZED 2 stereo camera** — 30 fps stereo video plus an onboard IMU, stored as a raw `.svo2` file
- **TI AWR1843BOOST radar** — range-Doppler heatmaps and detected points, roughly 15 Hz
- **Xsens MTi-8 IMU** — 200 Hz inertial data plus GNSS-fused position, stored as HDF5
- **Ouster OS-0-64 lidar** — 20 Hz point clouds, stored as a raw `.pcap` capture
- **u-blox F9P GNSS** — a raw NMEA log (often empty in practice; GNSS position is instead read from the MTi's internal sensor fusion output)

Because nothing is synchronized during recording, the application's core job is **nearest-timestamp alignment**: it treats the camera as the master clock and, for every frame, looks up the closest sample from every other sensor by comparing nanosecond timestamps. The result is a single window where a camera frame, its matching radar heatmap, lidar scan, IMU samples, and map position are all shown together, scrubbable like a video.

## Session & data loading

`session.py` defines `RecordingSession`, the object that loads one session directory into memory when the user picks it from the session dropdown. `RecordingSession.load()`:

1. Reads `manifests/dataset_manifest.json` (camera model, fps, resolution, sensor models) and the newest `metadata/session_manifest_*.json` (session id, start/stop time, duration).
2. Reads every parquet part-file under `metadata/frames/`, `metadata/radar/`, `metadata/events/`, `metadata/sessions/`, and `metadata/gnss/`, concatenating and sorting each by `timestamp_ns` into a pandas DataFrame.
3. Opens the session's `sensors/*.h5` file and loads the Xsens MTi-8 arrays (timestamp, acceleration, angular velocity, velocity, latitude/longitude/altitude, temperature, quaternion) into an `MtiData` object, which also exposes `nearest_index(timestamp_ns)` — a binary-search lookup used everywhere IMU data needs to be matched to a given moment.
4. If `raw/lidar.pcap` and `raw/lidar_0.json` both exist, builds a `LidarReader` (see Lidar subsystem below).

The camera itself is loaded separately by `main_window.py` via `SvoReader` (see Camera subsystem), since decoding the `.svo2` file needs the ZED SDK and is comparatively expensive.

Radar rows and their heatmap/point files are not preloaded; `RecordingSession.nearest_radar_row()`, `load_radar_heatmap()`, and `load_radar_points()` look them up and read them from disk on demand, once per playback frame.

## Camera subsystem

`svo_reader.py` defines `SvoReader`, which wraps the ZED SDK (`pyzed`) to make a `.svo2` recording seekable by timestamp rather than only by its own internal frame index.

**The problem it solves:** the dataset's `metadata/frames` parquet timestamps don't necessarily line up with the SVO file's internal frame numbering. So on first open of a session, `SvoReader` scans the entire `.svo2` file once, grabbing every frame to record its `IMAGE` timestamp *and* (in the same pass) the ZED's onboard IMU acceleration and angular velocity for that frame via `get_sensors_data(..., TIME_REFERENCE.IMAGE)`. This builds three parallel arrays indexed by SVO frame number: `timestamps_ns`, `zed_accel`, `zed_gyro`.

That scan is cached to `<recording>.frame_index.json` next to the `.svo2` file (a \~35 second one-time cost per session; instant afterward from cache). The cache is versioned, so changing what gets captured during the scan invalidates old caches automatically.

At playback time, `nearest_frame_index(timestamp_ns)` binary-searches `timestamps_ns` to find the SVO frame closest to the current master-clock timestamp, and `read_frame(index)` seeks the camera to that frame and retrieves both `LEFT` and `RIGHT` views as RGB arrays. The main window displays both simultaneously, stacked vertically.

## Radar subsystem

Each radar frame in `metadata/radar` points at two files: a range-Doppler heatmap (`.npy`, shape `[range_bins, doppler_bins]`) and a points file (`.json`, a list of detections with `x`, `y`, `z`, `velocity_mps`).

At each playback frame, `RecordingSession.nearest_radar_row(timestamp_ns)` finds the radar row closest to the current master-clock timestamp and reports the sync gap (`delta_ns`). The main window then:

- Loads and displays the heatmap in a `pyqtgraph.ImageView` (auto-ranged once on first load, so it fits the panel; not re-ranged every frame so the scale stays stable while scrubbing).
- Loads the points and plots them as a top-down `x`/`y` scatter (`y` = forward distance in meters), titled with the current sync delta and detection count so a stale match is visible at a glance.

## IMU subsystems

There are two independent IMUs in this dataset, shown as separate panels since they come from different hardware and different code paths:

- **MTi-8 (`sensors/*.h5`)** — 200 Hz dedicated IMU, loaded once as full arrays by `RecordingSession`. Plotted as full acceleration (x/y/z) and angular velocity (x/y/z) strip charts spanning the whole session, each with a moving vertical cursor line placed at `MtiData.nearest_index(timestamp_ns)`. The MTi also carries fused latitude/longitude, used for the GNSS/map panels below.
- **ZED 2 onboard IMU** — read directly from the `.svo2` recording (see Camera subsystem), one sample per camera frame rather than at the sensor's native rate. Plotted the same way, with its own cursor driven by the camera's own frame index rather than a separate nearest-neighbor search, since it's already 1:1 with the camera timeline.

If a recording's SVO2 doesn't contain onboard IMU data (`SvoReader.has_zed_imu` is `False`), the ZED IMU panels show “ZED onboard IMU not available in this recording” instead of an empty plot.

## Lidar subsystem

`lidar_reader.py` defines `LidarReader`, which parses an Ouster `raw/lidar.pcap` capture using its matching `raw/lidar_0.json` sensor metadata (via the `ouster-sdk` package).

**Sync problem it solves:** the Ouster's own per-column timestamps are sensor-clock, not host time (this unit is configured with `TIME_FROM_INTERNAL_OSC`), so they can't be compared directly to the dataset's other host-epoch timestamps. Each UDP packet's *host receive time* is available instead, via `LidarScan.packet_timestamp`, and that value is already a host-epoch nanosecond timestamp — the same clock the camera, radar, and IMU timestamps use. `LidarReader` uses the mean of each scan's `packet_timestamp` array as that scan's timestamp.

**Loading:** on session load, `LidarReader` iterates the whole pcap once (fast — well under a second for a typical session), converting each scan to XYZ points via `ouster.sdk.core.XYZLut` and keeping every 2nd row / 4th column to keep the full point-cloud cache small enough to hold in memory (roughly 15–20MB for a \~350-scan session) rather than re-parsing the pcap on every seek.

**Playback:** `nearest_index(timestamp_ns)` finds the closest scan to the current master-clock time; the main window renders it as a top-down `x`/`y` scatter colored by height (`z`), titled with the sync delta and point count. If `ouster-sdk` isn't installed or the pcap/json files are missing, the panel reports why instead of failing silently.

## GNSS & map

The raw `raw/gnss.nmea.log` is often empty on this hardware setup, so both GNSS displays instead read latitude/longitude straight from the MTi-8's own fused position output (loaded as part of `MtiData`).

- **GNSS track plot** — a `pyqtgraph` scatter of the full lon/lat track for the session, with a highlighted current-position marker driven by the same `MtiData.nearest_index()` used for the IMU cursors. If every sample is `(0, 0)` or non-finite (no fix at all in that recording), the panel says so instead of showing an empty plot.
- **Map panel (`map_view.py`)** — a `MapView` widget (`QWebEngineView` running a small embedded Leaflet page against live OpenStreetMap tiles) added below the track plot. On session load, the full valid lat/lon track is sent to the page as a polyline and the view is fit to its bounds; on every playback frame, the current position updates a marker on the map. It requires internet access to fetch OSM tiles.

## Playback & transport controls

The camera's frame table (`metadata/frames`, sorted by timestamp) is the master clock: `current_index` is always an index into this table, and everything else is looked up relative to the timestamp at that index.

- **Start** — jumps to frame 0.
- **Step ◀ / Step ▶** — move one frame at a time.
- **Play / Pause** — a `QTimer` advances `current_index` by one at an interval of `1000 / camera_fps / speed` ms, so playback speed scales with the manifest's actual recording fps; it stops automatically at the last frame.
- **Speed** — 0.25x–4x, changeable during playback (restarts the timer at the new interval).
- **Scrubber slider** — spans `0..num_frames-1`; dragging it seeks directly, same as stepping.
- **Events list** — populated from `metadata/events` (e.g. `RECORDING_START`, sensor drop counters); double-clicking an event seeks to the nearest camera frame at or after that event's timestamp.

Every seek calls `_update_frame()`, which re-renders all panels (camera, radar, lidar, both IMUs, GNSS, map) for the new timestamp in one pass.

## File & module reference

| File | Role |
| --- | --- |
| `main.py` | Entry point; creates the `QApplication` and `MainWindow`, defaulting to `output_multi-record/output` if no path is given on the command line |
| `main_window.py` | The `MainWindow` UI — layout, transport controls, session switching, and all per-frame rendering (`_render_camera`, `_render_radar`, `_render_lidar`, `_render_imu_cursor`) |
| `session.py` | `RecordingSession` — loads one session's manifests, parquet metadata, and MTi HDF5 data; wires in the `LidarReader` |
| `svo_reader.py` | `SvoReader` — seekable, timestamp-indexed access to a `.svo2` recording, plus the ZED onboard IMU capture and its on-disk cache |
| `lidar_reader.py` | `LidarReader` — parses an Ouster `.pcap` + metadata `.json` into an in-memory, timestamp-indexed point-cloud cache |
| `map_view.py` | `MapView` — the embedded Leaflet/OpenStreetMap panel showing the driven GNSS route and live position |
| `requirements.txt` | `PySide6`, `pyqtgraph`, `numpy`, `pandas`, `pyarrow`, `h5py`, `opencv-python-headless`, `ouster-sdk` |

### Expected on-disk layout per session

```
output/<session_id>/
  manifests/dataset_manifest.json
  metadata/session_manifest_*.json, frames/, radar/, events/, sessions/, gnss/  (parquet, date=partitioned)
  sensors/*.h5                  (MTi-8)
  radar/points/*.json, radar/heatmap/*.npy
  raw/recording.svo2            (ZED camera + onboard IMU)
  raw/lidar.pcap, raw/lidar_0.json   (Ouster)
  raw/gnss.nmea.log, raw/mti.xbus, raw/radar.bag   (raw logs, not read directly by the inspector)
```
