"""Thin CLI entry point for the standalone radar diagnostic dashboard.

Usage:
    python tests/radar_dashboard_runner.py [dataset_root] [session_dir]

With no arguments, launches the dashboard against the default
output_multi-record dataset root and shows the session selector.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from radar_dashboard import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
