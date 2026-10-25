"""Hand-engineered per-image scalar features, for the physics-honest Phase 14 baseline.

Analogous to how the existing tabular dataset's own 16 features (hep_quant.data.hls4ml_jets)
are hand-engineered substructure observables, not raw pixels -- this gives the classical
baseline something comparably informed to work with, rather than comparing a CNN only against
a "no physics knowledge" flattened-pixel control.

Pre-declared in reports/phase14_baseline_predeclaration.md before any training.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "total_energy", "nonzero_pixel_count",
    "centroid_row", "centroid_col",
    "spread_row", "spread_col",
    "max_pixel_energy",
]


def extract_image_features(images: np.ndarray) -> np.ndarray:
    """images: (N, H, W) -> (N, len(FEATURE_NAMES)) float64.

    Energy-weighted centroid/spread follow the same convention as a jet's own physical
    centroid/width: weight each pixel's (row, col) position by its energy, mirroring how the
    underlying jet-image literature defines a jet's centroid in (eta, phi). Images with zero
    total energy (should not occur for a real jet, but handled rather than dividing by zero)
    get centroid/spread set to the image center / zero.
    """
    _, h, w = images.shape
    rows, cols = np.indices((h, w))

    total_energy = images.sum(axis=(1, 2))
    nonzero_count = (images > 0).sum(axis=(1, 2)).astype(np.float64)
    max_energy = images.max(axis=(1, 2))

    safe_total = np.where(total_energy > 0, total_energy, 1.0)
    centroid_row = (images * rows).sum(axis=(1, 2)) / safe_total
    centroid_col = (images * cols).sum(axis=(1, 2)) / safe_total
    centroid_row = np.where(total_energy > 0, centroid_row, h / 2.0)
    centroid_col = np.where(total_energy > 0, centroid_col, w / 2.0)

    dr = rows[None, :, :] - centroid_row[:, None, None]
    dc = cols[None, :, :] - centroid_col[:, None, None]
    spread_row = (images * dr**2).sum(axis=(1, 2)) / safe_total
    spread_col = (images * dc**2).sum(axis=(1, 2)) / safe_total
    spread_row = np.where(total_energy > 0, spread_row, 0.0)
    spread_col = np.where(total_energy > 0, spread_col, 0.0)

    return np.stack(
        [total_energy, nonzero_count, centroid_row, centroid_col, spread_row, spread_col, max_energy],
        axis=1,
    )
