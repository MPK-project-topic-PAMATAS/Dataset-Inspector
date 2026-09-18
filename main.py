#!/usr/bin/env python3
"""Entry point for the Dataset Inspector GUI player.

Usage:
    python main.py [path/to/output_multi-record/output]
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is None:
        default = Path(__file__).parent / "output_multi-record" / "output"
        if default.exists():
            root = default
    win = MainWindow(dataset_root=root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
