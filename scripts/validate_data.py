#!/usr/bin/env python
"""Thin wrapper around `python -m hep_quant validate-data`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.__main__ import main

if __name__ == "__main__":
    sys.exit(main(["validate-data"] + sys.argv[1:]))
