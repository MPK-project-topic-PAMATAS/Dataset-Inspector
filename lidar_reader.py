"""Loads an Ouster .pcap + metadata .json recording into an in-memory,
timestamp-indexed point cloud cache for fast random-access playback.

The lidar's own per-column timestamps are sensor-clock, not host time (this
sensor is configured with TIME_FROM_INTERNAL_OSC), so they can't be compared
to the dataset's host-epoch timestamps directly. Each UDP packet's host
receive time is available instead (LidarPacket/LidarScan.packet_timestamp),
captured by the recorder when it wrote the pcap, and that value already is a
host-epoch nanosecond timestamp comparable to metadata/frames' timestamp_ns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import ouster.sdk as ouster_sdk
    import ouster.sdk.core as ouster_core
except ImportError:
    ouster_sdk = None
    ouster_core = None

# Every 2nd row / 4th column keeps the point cloud light enough to hold every
# scan of a session in memory (~15-20MB for a typical ~350-scan recording)
# while still being dense enough for a visual inspector.
_ROW_STRIDE = 2
_COL_STRIDE = 4


class LidarReader:
    def __init__(self, pcap_path: Path, meta_path: Path):
        if ouster_sdk is None:
            raise RuntimeError("ouster-sdk is not installed; lidar playback unavailable")
        self.pcap_path = Path(pcap_path)
        self.meta_path = Path(meta_path)

        source = ouster_sdk.open_source(str(self.pcap_path), meta=[str(self.meta_path)], collate=True)
        info = source.sensor_info[0]
        lut = ouster_core.XYZLut(info)

        timestamps = []
        points = []
        for frame_set in source:
            scan = frame_set[0]
            if scan is None:
                continue
            xyz = lut(scan)[::_ROW_STRIDE, ::_COL_STRIDE, :].reshape(-1, 3).astype(np.float32)
            valid = ~np.all(xyz == 0, axis=1)
            points.append(xyz[valid])
            timestamps.append(float(np.mean(scan.packet_timestamp)))

        self.timestamps_ns = np.asarray(timestamps, dtype=np.int64)
        self.points = points
        self.num_scans = len(points)

    def nearest_index(self, timestamp_ns: int):
        if self.num_scans == 0:
            return None, None
        idx = int(np.searchsorted(self.timestamps_ns, timestamp_ns))
        if idx <= 0:
            idx = 0
        elif idx >= self.num_scans:
            idx = self.num_scans - 1
        else:
            before, after = self.timestamps_ns[idx - 1], self.timestamps_ns[idx]
            idx = idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx
        return idx, int(self.timestamps_ns[idx]) - int(timestamp_ns)

    def points_at(self, index: int) -> np.ndarray:
        if index is None or not (0 <= index < self.num_scans):
            return np.zeros((0, 3), dtype=np.float32)
        return self.points[index]
