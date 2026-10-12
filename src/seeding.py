"""Random-state initialisation and reproducible CPU execution."""
from __future__ import annotations

import os
import random

import numpy as np

def apply_thread_limits() -> None:

    from experiment_config import SINGLE_THREAD_ENV

    for key, value in SINGLE_THREAD_ENV.items():
        os.environ[key] = value

    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except (RuntimeError, AttributeError):
            pass
    except ImportError:
        pass

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def rng_for(seed: int, salt: str = "") -> random.Random:

    if not salt:
        return random.Random(seed)
    mixed = f"{seed}::{salt}"
    return random.Random(int.from_bytes(mixed.encode("utf-8"), "little") % (2**63))
