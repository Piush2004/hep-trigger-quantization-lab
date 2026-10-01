"""Gate 0: environment sanity checks."""
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
import h5py


def test_numpy_basic_op():
    assert np.array([1, 2, 3]).sum() == 6


def test_pandas_basic_op():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert df["a"].sum() == 6


def test_sklearn_importable():
    assert sklearn.__version__ is not None


def test_torch_basic_op():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(x.sum(), torch.tensor(6.0))


def test_torch_forward_pass():
    layer = torch.nn.Linear(4, 2)
    out = layer(torch.randn(5, 4))
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


def test_yaml_roundtrip():
    data = {"a": 1, "b": [1, 2, 3]}
    assert yaml.safe_load(yaml.dump(data)) == data


def test_h5py_roundtrip(tmp_path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(10))
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["x"][:], np.arange(10))
