"""Deterministic seeding across Python, NumPy, and PyTorch."""
import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set the seed for every RNG this project touches.

    Sets Python's `random`, NumPy, PyTorch (CPU and CUDA), and requests
    cuDNN determinism. GPU convolution/reduction ops can still introduce
    small nondeterminism depending on the driver/kernel selected; this is
    documented in reports/reproducibility_report.md rather than hidden.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
