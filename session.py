"""Loads one recording session directory (e.g. output/20260914_070614) into memory."""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lidar_reader import LidarReader


def _read_parquet_dir(base: Path, name: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(base / name / "date=*" / "*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "timestamp_ns" in df.columns:
        df = df.sort_values("timestamp_ns").reset_index(drop=True)
    return df


@dataclass
class MtiData:
    timestamp_ns: np.ndarray
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    velocity: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    altitude_msl: np.ndarray
    temperature: np.ndarray
    quaternion_raw: np.ndarray

    @property
    def n(self) -> int:
        return len(self.timestamp_ns)

    def nearest_index(self, timestamp_ns: int) -> int:
        idx = int(np.searchsorted(self.timestamp_ns, timestamp_ns))
        if idx <= 0:
            return 0
        if idx >= self.n:
            return self.n - 1
        before, after = self.timestamp_ns[idx - 1], self.timestamp_ns[idx]
        return idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx


@dataclass
class RecordingSession:
    path: Path
    manifest: dict = field(default_factory=dict)
    session_manifest: dict = field(default_factory=dict)
    frames: pd.DataFrame = field(default_factory=pd.DataFrame)
    radar: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    sessions_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    gnss: pd.DataFrame = field(default_factory=pd.DataFrame)
    mti: MtiData | None = None
    lidar: LidarReader | None = None
    lidar_error: str | None = None

    @property
    def session_id(self) -> str:
        return self.path.name

    @property
    def svo_path(self) -> Path:
        return self.path / "raw" / "recording.svo2"

    @classmethod
    def load(cls, session_dir: Path) -> "RecordingSession":
        session_dir = Path(session_dir)
        manifest_path = session_dir / "manifests" / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        session_manifest = {}
        for f in sorted((session_dir / "metadata").glob("session_manifest_*.json")):
            session_manifest = json.loads(f.read_text())
            break

        meta = session_dir / "metadata"
        frames = _read_parquet_dir(meta, "frames")
        radar = _read_parquet_dir(meta, "radar")
        events = _read_parquet_dir(meta, "events")
        sessions_table = _read_parquet_dir(meta, "sessions")
        gnss = _read_parquet_dir(meta, "gnss")

        mti = None
        h5_files = sorted((session_dir / "sensors").glob("*.h5"))
        if h5_files:
            with h5py.File(h5_files[0], "r") as f:
                g = f["mti"]
                mti = MtiData(
                    timestamp_ns=g["timestamp_ns"][:],
                    acceleration=g["acceleration"][:],
                    angular_velocity=g["angular_velocity"][:],
                    velocity=g["velocity"][:],
                    latitude=g["latitude"][:],
                    longitude=g["longitude"][:],
                    altitude_msl=g["altitude_msl"][:],
                    temperature=g["temperature"][:],
                    quaternion_raw=g["quaternion_raw"][:],
                )

        lidar = None
        lidar_error = None
        pcap_path = session_dir / "raw" / "lidar.pcap"
        lidar_meta_path = session_dir / "raw" / "lidar_0.json"
        if pcap_path.exists() and lidar_meta_path.exists():
            try:
                lidar = LidarReader(pcap_path, lidar_meta_path)
            except Exception as e:
                lidar_error = str(e)

        return cls(
            path=session_dir,
            manifest=manifest,
            session_manifest=session_manifest,
            frames=frames,
            radar=radar,
            events=events,
            sessions_table=sessions_table,
            gnss=gnss,
            mti=mti,
            lidar=lidar,
            lidar_error=lidar_error,
        )

    def nearest_radar_row(self, timestamp_ns: int):
        if self.radar.empty:
            return None, None
        ts = self.radar["timestamp_ns"].to_numpy()
        idx = int(np.searchsorted(ts, timestamp_ns))
        if idx <= 0:
            idx = 0
        elif idx >= len(ts):
            idx = len(ts) - 1
        else:
            before, after = ts[idx - 1], ts[idx]
            idx = idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx
        row = self.radar.iloc[idx]
        return row, int(row["timestamp_ns"]) - int(timestamp_ns)

    def load_radar_heatmap(self, row) -> np.ndarray | None:
        p = row.get("heatmap_path") if row is not None else None
        if not p:
            return None
        full = self.path / p
        return np.load(full) if full.exists() else None

    def load_radar_points(self, row) -> list[dict]:
        p = row.get("points_path") if row is not None else None
        if not p:
            return []
        full = self.path / p
        if not full.exists():
            return []
        data = json.loads(full.read_text())
        return data.get("points", [])
