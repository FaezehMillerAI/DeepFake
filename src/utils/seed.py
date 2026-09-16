"""Global seeding. Call once, first thing, in every entry point."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy and torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Some ops have no deterministic kernel; warn rather than crash.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                pass
    except ImportError:
        pass

    return seed
