#!/usr/bin/env python3
"""Run from an unpacked archive without requiring editable installation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from symm_world.cli import main
if __name__ == "__main__":
    main()
