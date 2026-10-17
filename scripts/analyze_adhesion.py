#!/usr/bin/env python3
"""Pipeline 2: two-colour crops of a mixed culture, for judging cell-cell adhesion.

Each field is segmented, the regions holding the most labelled cells are
selected, every fluorescence channel is aligned to the phase image separately
within each crop, and the crops are written out as merged images plus a QC
panel. Cells carrying both colours are counted, which is the number to compare
between a mixed culture and its single-strain controls.

Example (red = nanobody strain, green = antigen strain):

    python scripts/analyze_adhesion.py \\
        --field mixed_100nM data/mix_phase.tif data/mix_red.tif data/mix_green.tif \\
        --field single_red   data/red_phase.tif data/red_only.tif \\
        --channel-names Nb2 Ag2 --channel-colors red green \\
        --pixel-size-um 0.1625 --out results/adhesion

A field given two images is treated as phase + one channel; three images as
phase + two channels. Outputs: crops.csv, cells.csv, one folder of figures per
field, and parameters.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecoli_imaging import crops as crops_mod
from ecoli_imaging import imageio, polarity, segmentation, viz
from ecoli_imaging.alignment import crop_with_offset, estimate_crop_shift
from ecoli_imaging.utils import DEFAULT_PIXEL_SIZE_UM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-colour crop extraction with per-channel alignment, for adhesion experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--field", nargs="+", action="append", required=True,
        metavar="NAME PHASE CH1 [CH2]",
        help="field name, phase-contrast image, and one or two fluorescence images. Repeat per field.",
    )
    parser.add_argument("--channel-names", nargs="+", default=["ch1", "ch2"],
                        help="names used in the output tables (default: ch1 ch2)")
    parser.add_argument("--channel-colors", nargs="+", default=["red", "green"],
                        choices=("red", "green", "magenta", "cyan", "yellow"),
                        help="display colour per channel (default: red green)")
    parser.add_argument("--out", default="results_adhesion", help="output directory")
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    parser.add_argument("--masks", nargs=2, action="append", metavar=("NAME", "MASKS"),
                        help="use a pre-computed label image for this field instead of Omnipose")
    parser.add_argument("--save-masks", action="store_true", help="write computed masks for reuse")

    group = parser.add_argument_group("crop selection")
    group.add_argument("--crop-size", type=int, default=512, help="side length of each crop in pixels")
    group.add_argument("--top-k", type=int, default=3, help="number of non-overlapping crops per field")
    group.add_argument("--stride", type=int, default=32, help="step of the sliding window")
    group.add_argument("--min-cells", type=int, default=10,
                       help="a crop position needs at least this many cells to be considered")

    group = parser.add_argument_group("alignment")
    group.add_argument("--max-shift", type=int, default=60,
                       help="maximum shift searched per channel, in pixels")
    group.add_argument("--fine-range", type=int, default=5,
                       help="fine search radius around the coarse solution")
    group.add_argument("--no-align", action="store_true", help="skip the per-channel correction")

    group = parser.add_argument_group("segmentation")
    group.add_argument("--model", default="bact_phase_omni")
    group.add_argument("--polarity-variant", default="both", choices=("inverted", "normal", "both"))
    group.add_argument("--scale", type=float, default=1.5)
    group.add_argument("--gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_lookup = {key: value for key, value in (args.masks or [])}

    seg_params = segmentation.SegmentationParams(
        model_type=args.model, scale=args.scale, use_gpu=args.gpu, polarity=args.polarity_variant)
    polarity_params = polarity.PolarityParams()

    crop_rows: list[dict] = []
    cell_rows: list[dict] = []

    for field in args.field:
        if len(field) < 3:
            raise SystemExit(f"--field needs a name, a phase image and at least one channel: {field}")
        name, phase_path, *channel_paths = field
        if len(channel_paths) > len(args.channel_names):
            raise SystemExit("more channels given than --channel-names provides")
        print(f"\n=== {name}")
        phase = imageio.load_image(phase_path)
        channels = {
            args.channel_names[i]: imageio.load_image(path)
            for i, path in enumerate(channel_paths)
        }
        colors = {args.channel_names[i]: args.channel_colors[i % len(args.channel_colors)]
                  for i in range(len(channel_paths))}
        for channel_name, image in channels.items():
            if image.shape != phase.shape:
                raise SystemExit(f"{name}: channel {channel_name} does not match the phase image size")

        if name in mask_lookup:
            labels, _ = segmentation.clean_masks(
                imageio.load_labels(mask_lookup[name]), seg_params.cleanup, args.pixel_size_um)
        else:
            brightest = max(channels.values(), key=lambda im: float(np.percentile(im, 99.9)))
            labels, _ = segmentation.segment_phase(phase, seg_params, args.pixel_size_um,
                                                   enrichment_image=brightest)
        if labels.max() == 0:
            print("  no cells found; skipping")
            continue
        print(f"  cells in field: {labels.max()}")

        field_dir = out_dir / name
        field_dir.mkdir(parents=True, exist_ok=True)
        if args.save_masks:
            imageio.save_labels(field_dir / "masks.npy", labels)

        cells, floors = crops_mod.evaluate_confinement(labels, channels)
        selected = crops_mod.find_disjoint_crops(
            cells, phase.shape, list(channels), size=args.crop_size,
            stride=args.stride, top_k=args.top_k, min_cells=args.min_cells)
        print(f"  selected {len(selected)} non-overlapping crops")

        viz.save_overview(field_dir / f"{name}_overview.png", phase, selected,
                          title=f"{name} - selected crops", pixel_size_um=args.pixel_size_um)

        for crop in selected:
            top, left, size = crop.top, crop.left, crop.size
            crop_phase = phase[top:top + size, left:left + size]
            crop_labels = labels[top:top + size, left:left + size]
            raw = {n: image[top:top + size, left:left + size] for n, image in channels.items()}

            aligned: dict[str, np.ndarray] = {}
            shifts: dict[str, tuple[int, int]] = {}
            scores: dict[str, tuple[float, float]] = {}
            for channel_name, image in channels.items():
                if args.no_align:
                    aligned[channel_name] = raw[channel_name]
                    shifts[channel_name] = (0, 0)
                    continue
                dy, dx, before, after = estimate_crop_shift(
                    crop_labels, raw[channel_name], args.max_shift, args.fine_range)
                # Cut the shifted crop out of the full image so no blank border appears.
                aligned[channel_name] = crop_with_offset(image, top, left, size, dy, dx)
                shifts[channel_name] = (dy, dx)
                scores[channel_name] = (before, after)
                print(f"    rank {crop.rank} {channel_name}: dy={dy:+d} dx={dx:+d} "
                      f"({before*100:.1f}% -> {after*100:.1f}% of the signal inside cells)")

            display_raw = {n: (raw[n], colors[n]) for n in channels}
            display_aligned = {n: (aligned[n], colors[n]) for n in channels}
            stem = f"{name}_rank{crop.rank}"

            viz.save_merged(field_dir / f"{stem}_merged.png", crop_phase, display_aligned,
                            crop_labels, args.pixel_size_um)
            if not args.no_align:
                viz.save_alignment_comparison(
                    field_dir / f"{stem}_alignment.png", crop_phase, display_raw, display_aligned,
                    crop_labels, shifts, scores, args.pixel_size_um,
                    title=f"{name} rank {crop.rank}")

            # Per-cell calls on the channel with the strongest in-mask signal,
            # so the QC panel says something about localization as well.
            cells_in_crop: list[dict] = []
            if channels:
                reference = max(aligned, key=lambda n: float(aligned[n][crop_labels > 0].sum()))
                measurement = segmentation.measurement_masks(crop_labels, 2)
                cells_in_crop = polarity.classify_cells(
                    aligned[reference], measurement, polarity_params,
                    rule="profile", pixel_size_um=args.pixel_size_um)
                for cell in cells_in_crop:
                    mask = crop_labels == cell["label"]
                    for channel_name in channels:
                        values = aligned[channel_name][mask]
                        cell[f"max_{channel_name}"] = float(values.max()) if values.size else 0.0
                        cell[f"positive_{channel_name}"] = bool(
                            values.size and values.max() >= floors[channel_name])
                    cell["double_positive"] = all(
                        cell.get(f"positive_{n}", False) for n in channels) and len(channels) > 1
                    cell.update(field=name, rank=crop.rank, reference_channel=reference,
                                crop_top=top, crop_left=left)
                cell_rows.extend(cells_in_crop)

            viz.save_qc_panel(field_dir / f"{stem}_qc.png", crop_phase, crop_labels,
                              display_aligned, cells_in_crop,
                              title=f"{name} - rank {crop.rank} "
                                    f"({crop.confinement_ratio*100:.0f}% of cells labelled)",
                              pixel_size_um=args.pixel_size_um)

            row = dict(field=name, rank=crop.rank, top=top, left=left, size=size,
                       confinement_ratio=crop.confinement_ratio, cells_total=crop.cells_total,
                       cells_any_channel=crop.cells_any_channel,
                       cells_all_channels=crop.cells_all_channels)
            for channel_name in channels:
                row[f"cells_{channel_name}"] = crop.cells_per_channel.get(channel_name, 0)
                row[f"shift_dy_{channel_name}"] = shifts[channel_name][0]
                row[f"shift_dx_{channel_name}"] = shifts[channel_name][1]
                if channel_name in scores:
                    row[f"in_mask_before_{channel_name}"] = scores[channel_name][0]
                    row[f"in_mask_after_{channel_name}"] = scores[channel_name][1]
            if cells_in_crop:
                row["double_positive_cells"] = int(sum(c.get("double_positive", False) for c in cells_in_crop))
            crop_rows.append(row)

    if not crop_rows:
        raise SystemExit("no crops were produced; check the input images")

    crops_table = pd.DataFrame(crop_rows)
    crops_table.to_csv(out_dir / "crops.csv", index=False)
    if cell_rows:
        pd.DataFrame(cell_rows).to_csv(out_dir / "cells.csv", index=False)

    # Per-field summary: the double-positive fraction is the adhesion read-out.
    if cell_rows:
        cells_table = pd.DataFrame(cell_rows)
        per_field = cells_table.groupby("field").agg(
            cells=("label", "size"),
            double_positive=("double_positive", "sum"),
        ).reset_index()
        per_field["double_positive_pct"] = 100.0 * per_field.double_positive / per_field.cells
        per_field.to_csv(out_dir / "summary.csv", index=False)
        print("\n" + per_field.to_string(index=False))

    json.dump(
        dict(command=" ".join(sys.argv), pixel_size_um=args.pixel_size_um,
             crop_size=args.crop_size, top_k=args.top_k, max_shift=args.max_shift,
             channel_names=args.channel_names, channel_colors=args.channel_colors,
             segmentation=asdict(seg_params) if not mask_lookup else "external masks"),
        open(out_dir / "parameters.json", "w"), indent=2, default=str,
    )
    print(f"\nwritten to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
