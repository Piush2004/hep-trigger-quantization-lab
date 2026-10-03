"""Dataset adapter for the hls4ml LHC jet-tagging benchmark (OpenML `hls4ml_lhc_jets_hlf`).

Keeps every dataset-specific assumption (expected schema, class set, feature
documentation) here rather than in the generic loader. `load_hls4ml_jets` wraps
the generic `load_openml` and then FAILS LOUDLY if the downloaded data does not
match the schema documented in reports/dataset_schema.md.

Primary sources for the documentation below:
  - Duarte et al., JINST 13 (2018) P07027, arXiv:1804.06913 (Section 2.2,
    "Input generation and features" and Table 1).
  - Zenodo record 10.5281/zenodo.3602260 (CC BY 4.0).
"""
from __future__ import annotations

from dataclasses import dataclass

from hep_quant.data.loader import HEPDataset, load_openml

OPENML_NAME = "hls4ml_lhc_jets_hlf"
OPENML_VERSION = 1

EXPECTED_CLASSES = ["g", "q", "t", "w", "z"]  # label-encoder (sorted) order


@dataclass(frozen=True)
class FeatureDoc:
    name: str
    meaning: str


# Meanings are limited to what the cited paper's Table 1 states (observable
# families and parameters). Where the paper only names the observable family,
# we say so rather than inventing a definition.
FEATURE_DOCS: list[FeatureDoc] = [
    FeatureDoc("zlogz", "Table 1 observable 'sum z log z' (definition of z not reviewed)"),
    FeatureDoc("c1_b0_mmdt", "Energy correlation function C1, beta=0 (Table 1: C1^{beta=0,1,2})"),
    FeatureDoc("c1_b1_mmdt", "Energy correlation function C1, beta=1"),
    FeatureDoc("c1_b2_mmdt", "Energy correlation function C1, beta=2"),
    FeatureDoc("c2_b1_mmdt", "Energy correlation function C2, beta=1 (Table 1: C2^{beta=1,2})"),
    FeatureDoc("c2_b2_mmdt", "Energy correlation function C2, beta=2"),
    FeatureDoc("d2_b1_mmdt", "Energy-correlation ratio D2, beta=1 (Table 1: D2^{beta=1,2})"),
    FeatureDoc("d2_b2_mmdt", "Energy-correlation ratio D2, beta=2"),
    FeatureDoc("d2_a1_b1_mmdt", "Generalized D2^{(alpha,beta)}, (alpha,beta)=(1,1) (Table 1: D2^{(alpha,beta)=(1,1),(1,2)})"),
    FeatureDoc("d2_a1_b2_mmdt", "Generalized D2^{(alpha,beta)}, (alpha,beta)=(1,2)"),
    FeatureDoc("m2_b1_mmdt", "Observable M2, beta=1 (Table 1: M2^{beta=1,2})"),
    FeatureDoc("m2_b2_mmdt", "Observable M2, beta=2"),
    FeatureDoc("n2_b1_mmdt", "Observable N2, beta=1 (Table 1: N2^{beta=1,2})"),
    FeatureDoc("n2_b2_mmdt", "Observable N2, beta=2"),
    FeatureDoc("mass_mmdt", "Table 1 observable 'mMDT' (jet mass after modified-mass-drop grooming, as inferred from the column name)"),
    FeatureDoc("multiplicity", "Table 1 observable 'Multiplicity'"),
]
EXPECTED_FEATURES = [f.name for f in FEATURE_DOCS]


class SchemaMismatchError(Exception):
    """Raised when downloaded data does not match the documented schema."""


def check_schema(ds: HEPDataset) -> None:
    if ds.feature_names != EXPECTED_FEATURES:
        raise SchemaMismatchError(
            f"Feature columns differ from documented schema.\n  got:      {ds.feature_names}\n  expected: {EXPECTED_FEATURES}"
        )
    if ds.class_names != EXPECTED_CLASSES:
        raise SchemaMismatchError(f"Classes {ds.class_names} != documented {EXPECTED_CLASSES}")
    if ds.X.shape[1] != len(EXPECTED_FEATURES):
        raise SchemaMismatchError(f"Expected {len(EXPECTED_FEATURES)} features, got {ds.X.shape[1]}")


def load_hls4ml_jets(subset_size: int | None = None, seed: int = 42, data_home: str = "data/raw/openml_cache") -> HEPDataset:
    ds = load_openml(name=OPENML_NAME, version=OPENML_VERSION, subset_size=subset_size, seed=seed, data_home=data_home)
    check_schema(ds)
    return ds
