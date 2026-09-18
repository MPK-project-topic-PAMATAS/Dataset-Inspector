"""Random-access reader for a ZED .svo2 recording, indexed by timestamp."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sl = None


class SvoReader:
    """Wraps pyzed.sl.Camera to allow seeking an SVO2 file by timestamp.

    On first open, scans the whole file once to record each internal frame's
    IMAGE timestamp, caching the result next to the .svo2 file so later opens
    are instant. This index is what lets the player map a dataset timestamp
    (from metadata/frames parquet) onto the correct SVO frame, since the two
    are not guaranteed to share the same numbering.
    """

    def __init__(self, svo_path: Path):
        if sl is None:
            raise RuntimeError("pyzed is not installed; camera playback unavailable")
        self.svo_path = Path(svo_path)
        self.cam = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(str(self.svo_path))
        init.svo_real_time_mode = False
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open {self.svo_path}: {status}")

        self.num_frames = self.cam.get_svo_number_of_frames()
        self.resolution = self.cam.get_camera_information().camera_configuration.resolution
        self.width = self.resolution.width
        self.height = self.resolution.height

        self._timestamps_ns, self._zed_accel, self._zed_gyro, self.has_zed_imu = self._load_or_build_index()

        self._left = sl.Mat()
        self._right = sl.Mat()
        self._depth = sl.Mat()

    _CACHE_VERSION = 2

    @property
    def _index_cache_path(self) -> Path:
        return self.svo_path.with_suffix(".frame_index.json")

    def _load_or_build_index(self):
        cache = self._index_cache_path
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                if data.get("num_frames") == self.num_frames and data.get("version") == self._CACHE_VERSION:
                    return (
                        np.asarray(data["timestamps_ns"], dtype=np.int64),
                        np.asarray(data["zed_accel"], dtype=np.float32),
                        np.asarray(data["zed_gyro"], dtype=np.float32),
                        bool(data["has_zed_imu"]),
                    )
            except Exception:
                pass

        timestamps = np.zeros(self.num_frames, dtype=np.int64)
        accel = np.zeros((self.num_frames, 3), dtype=np.float32)
        gyro = np.zeros((self.num_frames, 3), dtype=np.float32)
        sensors = sl.SensorsData()
        imu_hits = 0
        for i in range(self.num_frames):
            self.cam.set_svo_position(i)
            if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
                timestamps[i] = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                if self.cam.get_sensors_data(sensors, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    imu = sensors.get_imu_data()
                    accel[i] = imu.get_linear_acceleration()
                    gyro[i] = imu.get_angular_velocity()
                    imu_hits += 1
            elif i > 0:
                timestamps[i] = timestamps[i - 1]
                accel[i] = accel[i - 1]
                gyro[i] = gyro[i - 1]

        has_imu = imu_hits > 0
        try:
            cache.write_text(json.dumps({
                "version": self._CACHE_VERSION,
                "num_frames": self.num_frames,
                "timestamps_ns": timestamps.tolist(),
                "zed_accel": accel.tolist(),
                "zed_gyro": gyro.tolist(),
                "has_zed_imu": has_imu,
            }))
        except OSError:
            pass
        return timestamps, accel, gyro, has_imu

    @property
    def timestamps_ns(self) -> np.ndarray:
        return self._timestamps_ns

    @property
    def zed_accel(self) -> np.ndarray:
        return self._zed_accel

    @property
    def zed_gyro(self) -> np.ndarray:
        return self._zed_gyro

    def nearest_frame_index(self, timestamp_ns: int) -> int:
        idx = int(np.searchsorted(self._timestamps_ns, timestamp_ns))
        if idx <= 0:
            return 0
        if idx >= self.num_frames:
            return self.num_frames - 1
        before, after = self._timestamps_ns[idx - 1], self._timestamps_ns[idx]
        return idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx

    def read_frame(self, index: int, want_depth: bool = False) -> dict:
        """Seek to an absolute SVO frame index and decode it.

        Returns dict with left/right (HxWx3 RGB uint8) and optionally depth
        (HxW float32 meters), plus the frame's own timestamp for display.
        """
        index = max(0, min(index, self.num_frames - 1))
        self.cam.set_svo_position(index)
        err = self.cam.grab()
        result = {"index": index, "timestamp_ns": int(self._timestamps_ns[index]), "ok": err == sl.ERROR_CODE.SUCCESS}
        if not result["ok"]:
            return result

        self.cam.retrieve_image(self._left, sl.VIEW.LEFT)
        self.cam.retrieve_image(self._right, sl.VIEW.RIGHT)
        result["left"] = self._to_rgb(self._left)
        result["right"] = self._to_rgb(self._right)
        if want_depth:
            self.cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            result["depth"] = self._depth.get_data().copy()
        return result

    @staticmethod
    def _to_rgb(mat) -> np.ndarray:
        arr = mat.get_data()
        return arr[:, :, [2, 1, 0]].copy()

    def close(self):
        self.cam.close()
