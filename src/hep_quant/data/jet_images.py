"""Loader for the hls4ml LHC jet-image dataset (particle-level companion to hls4ml_lhc_jets_hlf).

Same underlying Pythia8 simulation and 5-class (g/q/t/w/z) labels as `hep_quant.data.hls4ml_jets`,
but with pre-rendered 100x100 calorimeter images instead of the 16 hand-engineered high-level
features. Source: Zenodo 10.5281/zenodo.3601436, "HLS4ML LHC Jet dataset (30 particles)"
(CC BY 4.0), supplement to Moreno et al. 2020 ("JEDI-net", arXiv:1908.05318) and Duarte et al.
2018 (arXiv:1804.06913, the same paper hls4ml_lhc_jets_hlf itself supplements).

Confirmed HDF5 schema (by direct inspection of a downloaded shard, not assumed): each shard file
has `jetImage` (N,100,100) float64 -- a single pre-rendered calorimeter-energy image per jet
(there are also separate `jetImageECAL`/`jetImageHCAL` channels, but ECAL is almost entirely
empty for these hadronic jets -- confirmed by direct inspection, ~4e-6 nonzero fraction on a
10,000-jet sample -- so this loader uses the single combined `jetImage` channel, matching the
single-channel convention in the jet-image literature (Cogan et al. 2015, arXiv:1407.5675; de
Oliveira et al. 2016, arXiv:1511.05190) rather than adding a near-empty second channel);
`jets` (N,59) with `jetFeatureNames` whose last 6 columns are the one-hot label
(g,q,w,z,t,undef -- undef never occurs in practice, confirmed empty across the full val split
and asserted, not assumed, for every split this loader reads).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

EXPECTED_LABEL_NAMES = ["j_g", "j_q", "j_w", "j_z", "j_t", "j_undef"]
CLASS_NAMES = ["g", "q", "w", "z", "t"]  # undef excluded -- never occurs, asserted in _check_schema


class SchemaMismatchError(Exception):
    """Raised when a shard file's schema doesn't match what this loader expects."""


@dataclass
class JetImageDataset:
    images: np.ndarray  # (N, H, W) float32, raw pixel (transverse-energy-like) intensity
    y: np.ndarray  # (N,) int64 label in [0, 4], indexing CLASS_NAMES
    class_names: list[str]
    source_files: list[str]


def _check_schema(f: h5py.File, path: str) -> None:
    if "jetImage" not in f or "jets" not in f or "jetFeatureNames" not in f:
        raise SchemaMismatchError(f"{path}: missing expected keys (jetImage/jets/jetFeatureNames)")
    names = [x.decode() if isinstance(x, bytes) else x for x in f["jetFeatureNames"][:]]
    if names[-6:] != EXPECTED_LABEL_NAMES:
        raise SchemaMismatchError(f"{path}: last 6 jetFeatureNames {names[-6:]} != expected {EXPECTED_LABEL_NAMES}")


def load_jet_images(
    data_dir: str,
    max_jets: int | None = None,
    seed: int = 42,
    per_shard_transform=None,
) -> JetImageDataset:
    """Load jet images + one-hot labels from every shard .h5 file under `data_dir`.

    Two-pass to stay cheap when subsampling: first reads only the small `jets` label array from
    every shard (global, unbiased random subsample indices are computed from this), then reads
    `jetImage` only for the selected rows -- never loads the (much larger) image array for a jet
    that would be discarded anyway. Without `max_jets`, every jet's image is read as normal.

    `per_shard_transform`, if given, is applied to each shard's raw (n_shard, 100, 100) float32
    image batch immediately after reading it, BEFORE accumulating into the final concatenated
    array -- e.g. `load_and_downsample_jet_images` passes `sum_pool` here so peak memory is one
    shard's worth of full-resolution images (~400MB for a 10,000-jet shard) plus the much
    smaller accumulated downsampled result, rather than materializing every selected jet's full
    100x100 image before downsampling (which, for the full 270,000-jet val split alone, would be
    ~10.8GB -- confirmed too large for this environment's ~10GB available RAM, checked directly).
    """
    files = sorted(glob.glob(str(Path(data_dir) / "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 shard files found under {data_dir}")

    per_file_labels, per_file_n = [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            _check_schema(f, fp)
            labels = f["jets"][:, 53:59]  # one-hot, columns matching EXPECTED_LABEL_NAMES order
        undef = labels[:, 5]
        if undef.sum() > 0:
            raise SchemaMismatchError(f"{fp}: {int(undef.sum())} 'undef'-labeled jets found (expected none)")
        per_file_labels.append(labels[:, :5].argmax(axis=1).astype(np.int64))
        per_file_n.append(len(labels))

    all_y = np.concatenate(per_file_labels, axis=0)
    total = len(all_y)
    if max_jets is not None and max_jets < total:
        rng = np.random.default_rng(seed)
        global_idx = np.sort(rng.choice(total, size=max_jets, replace=False))
    else:
        global_idx = np.arange(total)

    # Map the chosen global indices back to (file, local-row-within-file) pairs.
    offsets = np.concatenate([[0], np.cumsum(per_file_n)])
    file_of_idx = np.searchsorted(offsets, global_idx, side="right") - 1
    local_idx = global_idx - offsets[file_of_idx]

    images_list, y_list = [], []
    for file_i, fp in enumerate(files):
        rows = local_idx[file_of_idx == file_i]
        if len(rows) == 0:
            continue
        with h5py.File(fp, "r") as f:
            batch = f["jetImage"][rows].astype(np.float32)
        if per_shard_transform is not None:
            batch = per_shard_transform(batch)
        images_list.append(batch)
        y_list.append(all_y[offsets[file_i] + rows])

    images = np.concatenate(images_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return JetImageDataset(images=images, y=y, class_names=CLASS_NAMES, source_files=files)


def load_and_downsample_jet_images(
    data_dir: str, target_size: int, max_jets: int | None = None, seed: int = 42
) -> JetImageDataset:
    """Compose `load_jet_images` + `sum_pool`. Kept separate from the raw loader (rather than
    baking downsampling in) so full-resolution 100x100 images stay available for anyone who
    wants them later, e.g. a full-resolution ablation. `target_size` must evenly divide the raw
    100x100 image size -- 25 (factor 4) and 20 (factor 5) are both clean and match the
    literature-standard jet-image size (Cogan et al. 2015; de Oliveira et al. 2016, both cited
    in this module's docstring); a target like 32 does NOT evenly divide 100 and is rejected by
    `sum_pool`, not silently rounded or cropped.
    """
    if 100 % target_size != 0:
        raise ValueError(f"target_size={target_size} must evenly divide the raw 100x100 image size")
    factor = 100 // target_size
    return load_jet_images(
        data_dir, max_jets=max_jets, seed=seed,
        per_shard_transform=lambda batch: sum_pool(batch, factor=factor),
    )


def sum_pool(images: np.ndarray, factor: int) -> np.ndarray:
    """Downsample (N,H,W) images by summing factor x factor blocks.

    Sum (not mean/max) pooling conserves total calorimeter energy per jet under downsampling --
    the physically meaningful choice for an energy-intensity image. `H` and `W` must be evenly
    divisible by `factor`.
    """
    n, h, w = images.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"image size {h}x{w} not divisible by factor {factor}")
    reshaped = images.reshape(n, h // factor, factor, w // factor, factor)
    return reshaped.sum(axis=(2, 4))
