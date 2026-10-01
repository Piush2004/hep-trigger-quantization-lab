#!/usr/bin/env python
"""Placeholder for real-HEP-dataset download.

This project currently runs on a synthetic development dataset (see
reports/dataset_card.md for why, and the integration plan for a real dataset).
There is deliberately no download logic here yet: adding one before a specific
dataset URL/format is decided would risk silently depending on an untested
or unlicensed data source. When a real dataset is selected, implement its
download+checksum verification here, writing the result into data/raw/
(gitignored) and pointing a config's `dataset.path` at it.
"""
import sys

MESSAGE = """\
No real-HEP-dataset download is configured yet.

This project validates its pipeline on a synthetic dataset by default
(hep_quant.data.synthetic). To run on real data:
  1. Obtain a HEP tabular dataset (e.g. ATLAS/CMS Open Data) in HDF5/CSV/Parquet.
  2. Place it under data/raw/ (gitignored).
  3. Point a config's `dataset.path` at it and re-run `validate-data`.

See reports/dataset_card.md for the full integration plan.
"""


def main() -> int:
    print(MESSAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
