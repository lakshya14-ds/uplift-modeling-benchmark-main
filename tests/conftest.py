"""Pytest configuration and test path setup."""

import sys
from pathlib import Path

# Add src/ to python path so tests can import incremental_impact cleanly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
