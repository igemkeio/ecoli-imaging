"""Small helpers shared by every module: pixel scaling, robust statistics, integer shifts."""

from __future__ import annotations

import numpy as np

# Pixel size of the microscope used in the iGEM Keio 2026 experiments
# (40x objective, 6.5 um camera pixel). Override it on the command line
# with --pixel-size-um whenever your own setup differs.
DEFAULT_PIXEL_SIZE_UM = 0.1625


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    """Return (median, robust sigma) of an array.

    Sigma is estimated from the median absolute deviation (MAD) so that a few
    bright outliers - dust, a neighbouring cell, a hot pixel - cannot inflate it.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.0, 1e-6
    med = float(np.median(v))
    sigma = 1.4826 * float(np.median(np.abs(v - med)))
    return med, max(sigma, 1e-6)


def shift_image(image: np.ndarray, dy: int, dx: int, fill: float | None = None) -> np.ndarray:
    """Translate an image by whole pixels without wrap-around.

    Vacated border pixels are filled with `fill` (the image median by default),
    so a shifted fluorescence image never gains a black frame that would bias
    later background estimates.
    """
    dy, dx = int(dy), int(dx)
    if fill is None:
        fill = float(np.median(image))
    out = np.full_like(image, fill)
    h, w = image.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    src_y, dst_y = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    src_x, dst_x = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[dst_y, dst_x] = image[src_y, src_x]
    return out


def normalize(image: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Scale an image to 0-1 using percentile clipping (for display only)."""
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def um_to_px(value_um: float, pixel_size_um: float) -> float:
    """Convert a length in micrometres to pixels."""
    return value_um / pixel_size_um


def px_to_um(value_px: float, pixel_size_um: float) -> float:
    """Convert a length in pixels to micrometres."""
    return value_px * pixel_size_um
