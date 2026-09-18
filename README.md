# Dataset Inspector

Desktop GUI for reviewing multi-sensor vehicle recordings (ZED stereo camera, TI radar, Xsens IMU, Ouster lidar, u-blox GNSS) after the fact — scrub a synchronized timeline and watch every sensor update together. See [`SYSTEM_FUNCTIONS.md`](SYSTEM_FUNCTIONS.md) for a full technical breakdown.

## Requirements

- Python 3.10+
- (Optional, for the camera panel only) [ZED SDK](https://www.stereolabs.com/developers/release/) with its Python API — `pyzed` isn't on PyPI, so it's installed separately after the SDK is set up. Everything else (radar, lidar, IMU, GNSS, map) works without it.

## Installation

Clone the repo, then run the one-liner for your OS from the project root. Both create a virtual environment and install all dependencies.

**macOS / Linux:**
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
py -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## Running

```bash
python main.py [path/to/output_multi-record/output]
```

If no path is given, it defaults to `output_multi-record/output` in the project directory if that exists. Remember to activate the virtual environment first (`source .venv/bin/activate` / `.\.venv\Scripts\Activate.ps1`) in new shells.

## Testing

```bash
pip install -r requirements.txt  # includes pytest, pytest-qt, psutil
pytest
```

## License

MIT — see [`LICENSE`](LICENSE).
