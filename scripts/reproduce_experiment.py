#!/usr/bin/env python
"""Thin wrapper: python -m hep_quant reproduce --experiment <ID> (see hep_quant/experiments/reproduce.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.__main__ import main

if __name__ == "__main__":
    sys.exit(main(["reproduce"] + sys.argv[1:]))
