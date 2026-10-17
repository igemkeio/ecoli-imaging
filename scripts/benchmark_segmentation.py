#!/usr/bin/env python3
"""Pipeline 3: classical thresholding against Omnipose, on your own images.

Produces the comparison figure behind our decision to use a learned model:
rows are the regions you point it at, columns are raw phase contrast, Otsu,
Sauvola, local texture and Omnipose. Run it before committing to a model - if
your fields are evenly focused, a threshold may be all you need.

Example:

    python scripts/benchmark_segmentation.py \\
        --phase data/field1_phase.tif --phase data/field2_phase.tif \\
        --crop 1736 0 --crop 782 534 --crop-size 300 \\
        --out results/segmentation_benchmark

Without --crop, regions are chosen automatically: the ones where the methods
disagree most, which is where the choice of method actually matters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecoli_imaging import classical, imageio, segmentation
from ecoli_imaging.utils import DEFAULT_PIXEL_SIZE_UM, normalize
from ecoli_imaging.viz import add_scale_bar

METHOD_LABELS = {
    "otsu": "Global Otsu",
    "sauvola": "Adaptive Sauvola",
    "texture": "Local texture",
    "omnipose": "Omnipose",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare classical thresholding with Omnipose on phase-contrast images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--phase", action="append", required=True,
                        help="phase-contrast image; repeat for several fields")
    parser.add_argument("--out", default="results_segmentation", help="output directory")
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    parser.add_argument("--crop", nargs=2, type=int, action="append", metavar=("TOP", "LEFT"),
                        help="region to show; repeat. Without it, regions are picked automatically")
    parser.add_argument("--crop-size", type=int, default=300)
    parser.add_argument("--n-auto-crops", type=int, default=3,
                        help="number of regions to pick automatically per image")
    parser.add_argument("--masks", action="append",
                        help="pre-computed Omnipose labels for the matching --phase image")
    parser.add_argument("--skip-omnipose", action="store_true",
                        help="compare the classical methods only (no model needed)")
    parser.add_argument("--model", default="bact_phase_omni")
    parser.add_argument("--gpu", action="store_true")
    return parser


def auto_crops(masks: dict[str, np.ndarray], shape: tuple[int, int], size: int,
               count: int) -> list[tuple[int, int]]:
    """Pick the regions where the methods disagree the most.

    Disagreement is measured per block as the spread between how much area each
    method marks as cell. Regions where every method agrees teach nothing.
    """
    height, width = shape
    step = size
    scored = []
    stacks = list(masks.values())
    for top in range(0, height - size + 1, step):
        for left in range(0, width - size + 1, step):
            coverages = [float((m[top:top + size, left:left + size] > 0).mean()) for m in stacks]
            if max(coverages) < 0.01:      # an empty patch of agar tells us nothing
                continue
            scored.append((max(coverages) - min(coverages), top, left))
    scored.sort(reverse=True)
    chosen: list[tuple[int, int]] = []
    for _, top, left in scored:
        if all(abs(top - t) >= size or abs(left - l) >= size for t, l in chosen):
            chosen.append((top, left))
        if len(chosen) >= count:
            break
    return chosen or [(0, 0)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = args.masks or []

    methods = list(classical.METHOD_NAMES)
    if not args.skip_omnipose:
        methods.append("omnipose")

    stats_rows = []
    for index, phase_path in enumerate(args.phase):
        name = Path(phase_path).stem
        print(f"[{index + 1}/{len(args.phase)}] {name}")
        phase = imageio.load_image(phase_path)

        masks: dict[str, np.ndarray] = {}
        for method in classical.METHOD_NAMES:
            masks[method] = classical.segment_classical(phase, method, args.pixel_size_um)
        if not args.skip_omnipose:
            if index < len(mask_paths):
                masks["omnipose"] = imageio.load_labels(mask_paths[index])
            else:
                params = segmentation.SegmentationParams(model_type=args.model, use_gpu=args.gpu,
                                                         polarity="both")
                masks["omnipose"], _ = segmentation.segment_phase(phase, params, args.pixel_size_um)

        for method, labels in masks.items():
            stats_rows.append(dict(image=phase_path, method=method,
                                   **classical.segmentation_stats(labels, args.pixel_size_um)))

        regions = args.crop or auto_crops(masks, phase.shape, args.crop_size, args.n_auto_crops)
        _render(out_dir / f"{name}_comparison.png", phase, masks, regions,
                args.crop_size, methods, args.pixel_size_um, name)

    table = pd.DataFrame(stats_rows)
    table.to_csv(out_dir / "segmentation_stats.csv", index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nwritten to {out_dir.resolve()}")
    return 0


def _render(path: Path, phase: np.ndarray, masks: dict[str, np.ndarray],
            regions: list[tuple[int, int]], size: int, methods: list[str],
            pixel_size_um: float, title: str) -> None:
    """One row per region, one column per method, raw image first."""
    columns = len(methods) + 1
    rows = len(regions)
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.3 * rows), squeeze=False)
    colours = {"otsu": (1.0, 0.25, 0.25), "sauvola": (1.0, 0.65, 0.1),
               "texture": (1.0, 0.25, 0.25), "omnipose": (0.2, 0.95, 0.4)}

    for r, (top, left) in enumerate(regions):
        crop_phase = normalize(phase[top:top + size, left:left + size])
        axes[r][0].imshow(crop_phase, cmap="gray")
        if r == 0:
            axes[r][0].set_title("Raw phase contrast", fontsize=12)
        axes[r][0].set_ylabel(f"y={top}, x={left}", fontsize=10)
        for c, method in enumerate(methods, start=1):
            crop_labels = masks[method][top:top + size, left:left + size]
            rgb = np.dstack([crop_phase] * 3)
            tint = np.array(colours.get(method, (1.0, 0.3, 0.3)))
            body = crop_labels > 0
            rgb[body] = 0.6 * rgb[body] + 0.4 * tint
            rgb[find_boundaries(crop_labels, mode="outer")] = tint
            axes[r][c].imshow(np.clip(rgb, 0, 1))
            if r == 0:
                axes[r][c].set_title(METHOD_LABELS.get(method, method), fontsize=12)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            add_scale_bar(ax, 5.0, pixel_size_um)
    fig.suptitle(f"Segmentation methods compared - {title}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
