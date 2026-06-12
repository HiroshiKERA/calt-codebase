"""Fix all random seeds for fully reproducible experiments."""

import os
import random

import numpy as np
import torch
from transformers import set_seed as hf_set_seed


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for Python, NumPy, PyTorch, and HuggingFace.

    Call this at the top of any training or generation script.

    Parameters
    ----------
    seed : int
        The seed value. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hf_set_seed(seed)
    # Required env var for deterministic CuBLAS on CUDA >= 10.2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
