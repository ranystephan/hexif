"""Small image-array normalization helpers."""

from __future__ import annotations

import numpy as np


def to_float01(array: np.ndarray) -> np.ndarray:
    """Convert an image array to float32 using its recorded numeric range."""
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    result = array.astype(np.float32, copy=False)
    if array.dtype in (np.uint16, np.int16) and result.max(initial=0.0) > 1.5:
        scale = float(np.percentile(result, 99.9))
        if scale <= 0:
            raise ValueError("cannot normalize a non-positive integer image")
        result = result / scale
    elif result.max(initial=0.0) > 1.5:
        result = result / 255.0
    return result
