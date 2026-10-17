"""Correcting the shift between the phase-contrast and the fluorescence image.

Phase and fluorescence frames are acquired one after the other and do not land
on exactly the same pixels. In our data the offset ranged from a few pixels to
more than 120 px (about 20 um), differed between fluorescence channels of the
same field, and varied by a few pixels across one field. Overlaying the images
as acquired therefore puts the signal outside the cell outlines.

Both functions here find the translation that maximises how much fluorescence
falls inside the cell masks. Use `align_field` for a whole field (global shift
plus a per-tile refinement) and `estimate_crop_shift` for one crop of a
multi-channel experiment (coarse binary search, then a fine search on the raw
image).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from .utils import robust_stats, shift_image


@dataclass
class AlignmentParams:
    """Settings for the whole-field alignment."""

    threshold_k: float = 5.0        # binarise at background median + k * robust sigma
    global_search: int = 200        # +- px for the global search
    tile: int = 512                 # tile size for the local refinement
    tile_search: int = 12           # +- px searched around the global shift per tile
    tile_min_signal_px: int = 40    # tiles with less signal keep the global shift
    outlier_tolerance_px: int = 6   # tile shifts deviating more than this from the 3x3 median are replaced


@dataclass
class AlignmentResult:
    """What `align_field` returns."""

    aligned: np.ndarray             # fluorescence after the per-tile correction
    aligned_global: np.ndarray      # fluorescence after the global shift only
    global_shift: tuple[int, int]
    grid: np.ndarray                # (ny, nx, 2) per-tile shift in pixels
    tile_used: np.ndarray           # (ny, nx) bool: tile had enough signal to be refined
    threshold: float
    background_median: float
    background_sigma: float
    inclusion_raw: float            # fraction of signal pixels inside masks, before correction
    inclusion_global: float
    inclusion_tile: float
    n_signal_px: int


def background_threshold(fluor: np.ndarray, labels: np.ndarray, k: float) -> tuple[float, float, float]:
    """Threshold separating signal from background, measured well away from cells."""
    outside = ~ndi.binary_dilation(labels > 0, iterations=6)
    median, sigma = robust_stats(fluor[outside][::7])
    return median + k * sigma, median, sigma


def inclusion(binary_signal: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the thresholded signal pixels that fall inside the masks."""
    total = binary_signal.sum()
    return float((binary_signal & mask).sum() / total) if total else float("nan")


def _best_shift(mask: np.ndarray, binary_signal: np.ndarray, search: int) -> tuple[int, int, float]:
    """Cross-correlation peak within +- search pixels.

    Returns (dy, dx, score) such that shifting the fluorescence image by
    (+dy, +dx) puts its signal on the masks.
    """
    corr = fftconvolve(mask.astype(np.float32), binary_signal[::-1, ::-1].astype(np.float32), mode="full")
    cy, cx = binary_signal.shape[0] - 1, binary_signal.shape[1] - 1
    window = corr[cy - search: cy + search + 1, cx - search: cx + search + 1]
    iy, ix = np.unravel_index(int(np.argmax(window)), window.shape)
    return iy - search, ix - search, float(window[iy, ix])


def align_field(fluor: np.ndarray, labels: np.ndarray,
                params: AlignmentParams | None = None) -> AlignmentResult:
    """Align a full field: one global shift, then a refinement per tile.

    The global step fixes the large offset between the two acquisitions; the
    tile step absorbs the residual few pixels that vary across the field.
    Tiles without enough signal keep the global shift, and a tile whose shift
    disagrees with its neighbours is replaced by their median, so one noisy
    tile cannot tear the image apart.
    """
    params = params or AlignmentParams()
    threshold, bg_median, bg_sigma = background_threshold(fluor, labels, params.threshold_k)
    binary = fluor > threshold
    binary = ndi.binary_opening(binary, structure=np.ones((2, 2)))  # drop isolated hot pixels
    mask = labels > 0

    gdy, gdx, _ = _best_shift(mask, binary, params.global_search)

    tile = params.tile
    height, width = fluor.shape
    ny, nx = max(height // tile, 1), max(width // tile, 1)
    grid = np.zeros((ny, nx, 2), dtype=int)
    used = np.zeros((ny, nx), dtype=bool)
    reach = params.tile_search
    shifted_binary = shift_image(binary.astype(np.uint8), gdy, gdx, 0).astype(bool)

    for i in range(ny):
        for j in range(nx):
            ys, xs = slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile)
            # Pad the tile by the search radius so signal can move into it.
            y0, y1 = max(i * tile - reach, 0), min((i + 1) * tile + reach, height)
            x0, x1 = max(j * tile - reach, 0), min((j + 1) * tile + reach, width)
            local = np.zeros_like(shifted_binary[y0:y1, x0:x1])
            local[(i * tile - y0):(i * tile - y0 + tile), (j * tile - x0):(j * tile - x0 + tile)] = shifted_binary[ys, xs]
            if local.sum() < params.tile_min_signal_px:
                grid[i, j] = (gdy, gdx)
                continue
            dy, dx, _ = _best_shift(mask[y0:y1, x0:x1], local, reach)
            grid[i, j] = (gdy + dy, gdx + dx)
            used[i, j] = True

    smoothed = grid.copy()
    for i in range(ny):
        for j in range(nx):
            if not used[i, j]:
                continue
            neighbours = [
                grid[a, b]
                for a in range(max(i - 1, 0), min(i + 2, ny))
                for b in range(max(j - 1, 0), min(j + 2, nx))
                if used[a, b] and (a, b) != (i, j)
            ]
            if len(neighbours) >= 2:
                median = np.median(neighbours, axis=0)
                if np.abs(grid[i, j] - median).max() > params.outlier_tolerance_px:
                    smoothed[i, j] = np.round(median).astype(int)
    grid = smoothed

    fill = float(bg_median)
    aligned = np.full_like(fluor, fill, dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            ys, xs = slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile)
            shifted = shift_image(fluor, int(grid[i, j, 0]), int(grid[i, j, 1]), fill)
            aligned[ys, xs] = shifted[ys, xs]
    # Tiles are a whole number of `tile` pixels; copy any remainder from the global shift.
    aligned_global = shift_image(fluor, gdy, gdx, fill)
    if ny * tile < height:
        aligned[ny * tile:, :] = aligned_global[ny * tile:, :]
    if nx * tile < width:
        aligned[:, nx * tile:] = aligned_global[:, nx * tile:]

    aligned_binary = ndi.binary_opening(aligned > threshold, structure=np.ones((2, 2)))
    return AlignmentResult(
        aligned=aligned,
        aligned_global=aligned_global,
        global_shift=(int(gdy), int(gdx)),
        grid=grid,
        tile_used=used,
        threshold=float(threshold),
        background_median=float(bg_median),
        background_sigma=float(bg_sigma),
        inclusion_raw=inclusion(binary, mask),
        inclusion_global=inclusion(shift_image(binary.astype(np.uint8), gdy, gdx, 0).astype(bool), mask),
        inclusion_tile=inclusion(aligned_binary, mask),
        n_signal_px=int(binary.sum()),
    )


def estimate_crop_shift(labels: np.ndarray, fluor: np.ndarray,
                        max_shift: int = 60, fine_range: int = 5) -> tuple[int, int, float, float]:
    """Coarse-to-fine shift for one crop and one fluorescence channel.

    Step 1 binarises the brightest signal and finds the correlation peak within
    +- max_shift px. Step 2 searches +- fine_range px around that solution on
    the raw image, maximising the fluorescence energy inside the masks.

    Returns (dy, dx, in_mask_fraction_before, in_mask_fraction_after). When the
    crop holds no signal above the noise floor the shift is (0, 0): a channel
    that is simply not expressed must not be dragged onto the cells.
    """
    background = labels == 0
    bg_values = fluor[background] if background.any() else fluor
    bg_median = float(np.median(bg_values))
    bg_std = float(np.std(bg_values))
    noise_floor = bg_median + 3.0 * bg_std

    excess = np.clip(fluor.astype(np.float32) - bg_median, 0.0, None)
    total_energy = float(excess.sum()) + 1e-6
    before = float(excess[labels > 0].sum()) / total_energy

    if float(fluor.max()) <= noise_floor or total_energy <= 1e-3:
        return 0, 0, before, before

    # Step 1: coarse search on the strongest signal only.
    threshold = max(noise_floor, float(np.percentile(fluor, 95.0)))
    binary = (fluor >= threshold).astype(np.float32)
    if binary.sum() < 20:
        binary = (fluor >= noise_floor).astype(np.float32)
    mask = (labels > 0).astype(np.float32)
    if binary.sum() < 10 or mask.sum() < 10:
        return 0, 0, before, before

    corr = fftconvolve(mask, binary[::-1, ::-1], mode="same")
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    y0, y1 = max(0, cy - max_shift), min(corr.shape[0], cy + max_shift + 1)
    x0, x1 = max(0, cx - max_shift), min(corr.shape[1], cx + max_shift + 1)
    iy, ix = np.unravel_index(int(np.argmax(corr[y0:y1, x0:x1])), (y1 - y0, x1 - x0))
    coarse_dy, coarse_dx = int((y0 + iy) - cy), int((x0 + ix) - cx)

    # Step 2: fine search on the raw image.
    best_dy, best_dx = coarse_dy, coarse_dx
    best_energy = float(shift_image(excess, coarse_dy, coarse_dx, 0.0)[labels > 0].sum())
    for dy in range(coarse_dy - fine_range, coarse_dy + fine_range + 1):
        for dx in range(coarse_dx - fine_range, coarse_dx + fine_range + 1):
            if abs(dy) > max_shift or abs(dx) > max_shift:
                continue
            energy = float(shift_image(excess, dy, dx, 0.0)[labels > 0].sum())
            if energy > best_energy:
                best_energy = energy
                best_dy, best_dx = dy, dx

    after = best_energy / total_energy
    return int(best_dy), int(best_dx), before, float(after)


def crop_with_offset(full_image: np.ndarray, top: int, left: int, size: int,
                     dy: int, dx: int) -> np.ndarray:
    """Cut a crop out of the full image already displaced by (dy, dx).

    Taking the pixels from the shifted position in the *full* image, rather
    than shifting a crop that has already been cut, keeps the crop free of the
    blank border a translation would otherwise leave.
    """
    height, width = full_image.shape
    y0, x0 = top - dy, left - dx
    out = np.full((size, size), float(np.median(full_image)), dtype=np.float32)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + size, height), min(x0 + size, width)
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = full_image[sy0:sy1, sx0:sx1]
    return out
