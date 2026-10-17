#!/usr/bin/env python3
"""Generate a small synthetic dataset, so the pipelines can be tried without real images.

It writes phase-contrast and fluorescence TIFFs for three mock induction levels
plus a two-colour mixed-culture field, and the ground-truth label images. The
fluorescence is deliberately shifted relative to the phase image by a different
amount in each field, which is what the alignment step has to recover.

    python scripts/make_demo_data.py --out demo_data

Then, without needing Omnipose installed (the masks are supplied):

    python scripts/analyze_polarity.py \\
        --field low  demo_data/low_phase.tif  demo_data/low_fluor.tif \\
        --field high demo_data/high_phase.tif demo_data/high_fluor.tif \\
        --masks low demo_data/low_masks.npy --masks high demo_data/high_masks.npy \\
        --out demo_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def draw_cell(canvas: np.ndarray, cy: float, cx: float, length: float, width: float,
              angle: float, value: float, label_image: np.ndarray | None = None,
              label: int = 0) -> np.ndarray:
    """Draw one rod-shaped cell (a capsule) and optionally record its label."""
    height, w = canvas.shape
    ys, xs = np.mgrid[0:height, 0:w]
    dy, dx = ys - cy, xs - cx
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    half = max(length / 2 - width / 2, 0.1)
    distance = np.hypot(np.clip(np.abs(along) - half, 0, None), across)
    body = distance <= width / 2
    canvas[body] = value
    if label_image is not None:
        label_image[body] = label
    return body


def make_field(rng: np.random.Generator, size: int = 512, n_cells: int = 45,
               expressing: float = 0.8, unipolar: float = 0.8,
               shift: tuple[int, int] = (7, -4), brightness: float = 1.0):
    """Build one field: phase image, fluorescence image, and ground-truth labels."""
    phase = np.full((size, size), 1000.0)
    labels = np.zeros((size, size), dtype=np.int32)
    fluor = np.zeros((size, size), dtype=np.float32)

    placed = 0
    attempts = 0
    while placed < n_cells and attempts < n_cells * 60:
        attempts += 1
        cy, cx = rng.uniform(40, size - 40, size=2)
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(17, 20)      # ~2.8-3.2 um at 0.1625 um/px
        width = rng.uniform(5.2, 5.9)     # fitted ellipse minor axis lands near 1.0 um
        probe = np.zeros_like(labels)
        body = draw_cell(np.zeros_like(phase), cy, cx, length + 8, width + 8, angle, 1.0, probe, 1)
        if (labels[body] > 0).any():
            continue
        placed += 1
        draw_cell(phase, cy, cx, length, width, angle, 760.0, labels, placed)

        if rng.random() < expressing:
            # A polar focus sits near one tip; a non-polar cell is filled evenly.
            peak = brightness * rng.uniform(900, 1600)
            if rng.random() < unipolar:
                offset = (length / 2 - width / 2) * rng.uniform(0.55, 0.95)
                sign = 1 if rng.random() < 0.5 else -1
                fy = cy + sign * offset * np.sin(angle)
                fx = cx + sign * offset * np.cos(angle)
                spot = np.exp(-(((np.mgrid[0:size, 0:size][0] - fy) ** 2
                                 + (np.mgrid[0:size, 0:size][1] - fx) ** 2) / (2 * 2.1 ** 2)))
                fluor += (peak * spot).astype(np.float32)
            else:
                cell = labels == placed
                fluor[cell] += peak * 0.35

    # The microscope: blur, a shift between the channels, background and noise.
    from scipy import ndimage as ndi

    fluor = ndi.gaussian_filter(fluor, 1.4)
    fluor = np.roll(np.roll(fluor, -shift[0], axis=0), -shift[1], axis=1)
    phase = ndi.gaussian_filter(phase, 1.1)
    phase += rng.normal(0, 12, phase.shape)
    fluor += 300 + rng.normal(0, 11, fluor.shape)
    return (np.clip(phase, 0, 65535).astype(np.uint16),
            np.clip(fluor, 0, 65535).astype(np.uint16),
            labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a synthetic demo dataset.")
    parser.add_argument("--out", default="demo_data")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Three induction levels: unipolar localization degrades as expression rises.
    for name, unipolar, brightness, shift in (("low", 0.92, 0.6, (7, -4)),
                                              ("mid", 0.65, 1.0, (-11, 9)),
                                              ("high", 0.25, 2.2, (3, 14))):
        phase, fluor, labels = make_field(rng, args.size, unipolar=unipolar,
                                          brightness=brightness, shift=shift)
        tifffile.imwrite(out / f"{name}_phase.tif", phase)
        tifffile.imwrite(out / f"{name}_fluor.tif", fluor)
        np.save(out / f"{name}_masks.npy", labels)
        print(f"{name}: {labels.max()} cells, applied shift {shift}")

    # A two-colour field for the adhesion pipeline: two populations, one per colour.
    phase, red, labels = make_field(rng, args.size, expressing=0.45, unipolar=0.1,
                                    brightness=1.4, shift=(9, 5))
    _, green, _ = make_field(rng, args.size, expressing=0.0)
    green = green.astype(np.float32)
    for value in range(1, labels.max() + 1):
        if value % 2 == 0:                       # half the cells carry the green label instead
            cell = labels == value
            green[cell] += 1500
            red = red.astype(np.float32)
            red[cell] *= 0.2
    from scipy import ndimage as ndi

    green = ndi.gaussian_filter(green, 1.4)
    green = np.roll(np.roll(green, -4, axis=0), -12, axis=1)
    tifffile.imwrite(out / "mixed_phase.tif", phase)
    tifffile.imwrite(out / "mixed_red.tif", np.clip(red, 0, 65535).astype(np.uint16))
    tifffile.imwrite(out / "mixed_green.tif", np.clip(green, 0, 65535).astype(np.uint16))
    np.save(out / "mixed_masks.npy", labels)
    print(f"mixed: {labels.max()} cells, two colours")
    print(f"\nwritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
