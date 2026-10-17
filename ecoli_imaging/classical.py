"""Classical thresholding, kept so the choice of a deep-learning model can be checked.

These are the three methods we tried before Omnipose. Running
`scripts/benchmark_segmentation.py` on your own images reproduces the
comparison figure and tells you whether your data need a learned model at all:
on a clean, evenly focused field, Otsu is perfectly adequate and far faster.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.measure import label as cc_label

from .utils import DEFAULT_PIXEL_SIZE_UM, normalize

METHOD_NAMES = ("otsu", "sauvola", "texture")


def flatten_background(image: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Remove slow illumination shading, which otherwise dominates a global threshold."""
    background = ndi.gaussian_filter(image.astype(np.float32), sigma=sigma)
    return image.astype(np.float32) - background


def segment_classical(image: np.ndarray, method: str = "otsu",
                      pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
                      min_area_um2: float = 0.4, window: int = 25) -> np.ndarray:
    """Segment a phase-contrast image with one classical method.

    otsu     : one global threshold for the whole image
    sauvola  : a threshold that follows local mean and standard deviation
    texture  : local intensity variance, which responds to the striped texture
               of a cell rather than to its absolute brightness

    All three return a label image, so their output can be compared with the
    model's directly.
    """
    flat = flatten_background(image)
    normed = normalize(flat)

    if method == "otsu":
        # Cells are darker than the background in phase contrast, so the dark side is kept.
        binary = normed < threshold_otsu(normed)
    elif method == "sauvola":
        binary = normed < threshold_sauvola(normed, window_size=window)
    elif method == "texture":
        mean = ndi.uniform_filter(normed, size=9)
        variance = ndi.uniform_filter(normed ** 2, size=9) - mean ** 2
        binary = variance > np.percentile(variance, 97.0)
        binary = ndi.binary_closing(binary, structure=np.ones((3, 3)))
        binary = ndi.binary_fill_holes(binary)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHOD_NAMES}")

    # Drop debris below the smallest plausible cell. Done by hand rather than
    # with remove_small_objects, whose size keyword changed between versions.
    min_px = max(int(min_area_um2 / pixel_size_um ** 2), 1)
    labels = cc_label(binary)
    areas = np.bincount(labels.ravel())
    too_small = np.flatnonzero(areas < min_px)
    if too_small.size:
        labels[np.isin(labels, too_small)] = 0
    return cc_label(labels > 0).astype(np.int32)


def segmentation_stats(labels: np.ndarray, pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM) -> dict:
    """Counts and median object size - enough to see a method failing."""
    from skimage.measure import regionprops

    props = regionprops(labels)
    areas = np.array([p.area for p in props], dtype=float) * pixel_size_um ** 2
    return {
        "n_objects": len(props),
        "median_area_um2": float(np.median(areas)) if areas.size else float("nan"),
        "covered_fraction": float((labels > 0).mean()),
    }
