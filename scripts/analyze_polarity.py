#!/usr/bin/env python3
"""Pipeline 1: is the fluorescence at one pole, and how does that change with induction?

For every field it segments the cells, corrects the shift between the phase and
the fluorescence image, selects normally grown cells by size, and calls each
cell unipolar / non-unipolar / no fluorescence. Fields that share a condition
label are pooled, so one run over an induction series produces the dose-response
figure directly.

Example (one field per IPTG concentration):

    python scripts/analyze_polarity.py \\
        --field "1 nM"   data/1nM_phase.tif   data/1nM_mcherry.tif \\
        --field "10 nM"  data/10nM_phase.tif  data/10nM_mcherry.tif \\
        --field "100 nM" data/100nM_phase.tif data/100nM_mcherry.tif \\
        --pixel-size-um 0.1625 --out results/popz

Give --field the same label twice to pool two fields of the same condition.
Outputs: cells.csv, fields.csv, summary.csv, parameters.json and one PNG per
field plus the summary figures.
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

from ecoli_imaging import imageio, polarity, segmentation, viz
from ecoli_imaging.alignment import AlignmentParams, align_field
from ecoli_imaging.utils import DEFAULT_PIXEL_SIZE_UM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-cell polar localization analysis across an induction series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--field", nargs=3, action="append", required=True,
        metavar=("LABEL", "PHASE", "FLUOR"),
        help="condition label, phase-contrast image, fluorescence image. Repeat for every field.",
    )
    parser.add_argument("--out", default="results_polarity", help="output directory")
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM,
                        help=f"microns per pixel (default {DEFAULT_PIXEL_SIZE_UM}, a 40x objective)")
    parser.add_argument("--masks", nargs=2, action="append", metavar=("LABEL_OR_PHASE", "MASKS"),
                        help="use a pre-computed label image instead of running Omnipose. "
                             "Match it either to the field label or to the phase file path.")
    parser.add_argument("--save-masks", action="store_true",
                        help="write the computed masks to <out>/masks/ for reuse with --masks")

    group = parser.add_argument_group("segmentation")
    group.add_argument("--model", default="bact_phase_omni", help="Omnipose model name")
    group.add_argument("--polarity-variant", default="both", choices=("inverted", "normal", "both"),
                       help="run the model on the inverted phase image, the normal one, or both "
                            "and combine them (default: both)")
    group.add_argument("--scale", type=float, default=1.5,
                       help="upscale factor before inference; raise it if your cells are thinner than ~6 px")
    group.add_argument("--gpu", action="store_true", help="run the model on the GPU")

    group = parser.add_argument_group("cell selection (micrometres)")
    group.add_argument("--length-um", nargs=2, type=float, default=(2.0, 4.0),
                       help="length window of the analysed cells (max Feret diameter)")
    group.add_argument("--width-um", nargs=2, type=float, default=(0.8, 1.2))
    group.add_argument("--area-um2", nargs=2, type=float, default=(1.4, 4.2))
    group.add_argument("--no-size-filter", action="store_true",
                       help="analyse every segmented cell, not only those inside the size window")

    group = parser.add_argument_group("alignment")
    group.add_argument("--global-search", type=int, default=200,
                       help="maximum global shift searched, in pixels")
    group.add_argument("--tile", type=int, default=512, help="tile size for the local refinement")
    group.add_argument("--tile-search", type=int, default=12,
                       help="pixels searched around the global shift within each tile")
    group.add_argument("--no-align", action="store_true",
                       help="skip the correction (only if your channels are already registered)")

    group = parser.add_argument_group("classification")
    group.add_argument("--rule", default="half", choices=("half", "profile"),
                       help="'half': unipolar when the signal stays in one half of the cell. "
                            "'profile': four-way call including bipolar (see the README caveat)")
    group.add_argument("--half-min", type=float, default=0.90,
                       help="fraction of the focus signal that must lie in one half for 'unipolar'")
    group.add_argument("--signal-k", type=float, default=3.0,
                       help="a pixel is signal above local background + k * sigma")
    group.add_argument("--focus-level", type=float, default=0.4,
                       help="focus pixels are those at this fraction of the cell's peak or above")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    mask_lookup = {key: value for key, value in (args.masks or [])}

    cleanup = segmentation.MaskCleanupParams(
        strict_len_um=tuple(args.length_um),
        strict_wid_um=tuple(args.width_um),
        strict_area_um2=tuple(args.area_um2),
    )
    seg_params = segmentation.SegmentationParams(
        model_type=args.model, scale=args.scale, use_gpu=args.gpu,
        polarity=args.polarity_variant, cleanup=cleanup,
    )
    align_params = AlignmentParams(global_search=args.global_search, tile=args.tile,
                                   tile_search=args.tile_search)
    polarity_params = polarity.PolarityParams(
        half_min=args.half_min, signal_k=args.signal_k, focus_level=args.focus_level,
    )

    cell_rows: list[dict] = []
    field_rows: list[dict] = []
    shift_grids: dict[str, np.ndarray] = {}

    for index, (label, phase_path, fluor_path) in enumerate(args.field, start=1):
        name = f"{label}_{index}"
        print(f"[{index}/{len(args.field)}] {label}: {phase_path}")
        phase = imageio.load_image(phase_path)
        fluor = imageio.load_image(fluor_path)
        if phase.shape != fluor.shape:
            raise SystemExit(f"{name}: phase {phase.shape} and fluorescence {fluor.shape} differ in size")

        mask_path = mask_lookup.get(label) or mask_lookup.get(phase_path)
        if mask_path:
            labels, seg_stats = segmentation.clean_masks(
                imageio.load_labels(mask_path), cleanup, args.pixel_size_um)
            seg_stats["segmentation_variant"] = "external"
        else:
            labels, seg_stats = segmentation.segment_phase(
                phase, seg_params, args.pixel_size_um, enrichment_image=fluor)
        if labels.max() == 0:
            print(f"  no cells found in {name}; skipping")
            continue
        if args.save_masks:
            imageio.save_labels(out_dir / "masks" / f"{name}.npy", labels)
        print(f"  cells after clean-up: {labels.max()}")

        if args.no_align:
            aligned = fluor
            field_info = dict(global_dy=0, global_dx=0, inclusion_raw=np.nan, inclusion_tile=np.nan)
        else:
            result = align_field(fluor, labels, align_params)
            aligned = result.aligned
            shift_grids[name] = result.grid
            field_info = dict(
                global_dy=result.global_shift[0], global_dx=result.global_shift[1],
                tile_dy_min=int(result.grid[..., 0].min()), tile_dy_max=int(result.grid[..., 0].max()),
                tile_dx_min=int(result.grid[..., 1].min()), tile_dx_max=int(result.grid[..., 1].max()),
                inclusion_raw=result.inclusion_raw, inclusion_global=result.inclusion_global,
                inclusion_tile=result.inclusion_tile, n_signal_px=result.n_signal_px,
            )
            print(f"  shift (dy, dx) = {result.global_shift}, "
                  f"signal inside masks {result.inclusion_raw*100:.1f}% -> {result.inclusion_tile*100:.1f}%")
            viz.save_alignment_comparison(
                out_dir / "figures" / f"{name}_alignment.png",
                phase, {"fluor": (fluor, "magenta")}, {"fluor": (aligned, "magenta")},
                labels, {"fluorescence": result.global_shift},
                {"fluorescence": (result.inclusion_raw, result.inclusion_tile)},
                args.pixel_size_um, title=f"{label} - shift correction",
            )

        measurement = segmentation.measurement_masks(labels, polarity_params.ring_inner_px - 1)
        cells = polarity.classify_cells(aligned, measurement, polarity_params,
                                        rule=args.rule, pixel_size_um=args.pixel_size_um)
        sizes = segmentation.size_table(labels, cleanup, args.pixel_size_um).set_index("label")
        for cell in cells:
            if cell["label"] in sizes.index:
                cell.update(sizes.loc[cell["label"]].to_dict())
            if args.no_size_filter:
                cell["strict_ok"] = True
            cell.update(condition=label, field=name, phase_image=phase_path, fluor_image=fluor_path)
        cell_rows.extend(cells)

        viz.save_classification_overlay(
            out_dir / "figures" / f"{name}_classification.png",
            phase, aligned, labels, cells, args.pixel_size_um,
            title=f"{label} - per-cell calls",
        )

        analysed = [c for c in cells if c.get("strict_ok", True)]
        field_rows.append(dict(
            field=name, condition=label, phase_image=phase_path, fluor_image=fluor_path,
            cells_detected=int(labels.max()), cells_analysed=len(analysed),
            segmentation=seg_stats.get("segmentation_variant", "omnipose"),
            merged_slivers=seg_stats.get("merged_slivers", 0),
            rejected_border=seg_stats.get("rejected_border", 0),
            rejected_size=seg_stats.get("rejected_size", 0),
            **field_info,
        ))

    if not cell_rows:
        raise SystemExit("no cells were analysed; check the input images")

    cells_table = pd.DataFrame(cell_rows)
    fields_table = pd.DataFrame(field_rows)
    summary = polarity.summarise(cell_rows, group_key="condition",
                                 only_strict=not args.no_size_filter)

    cells_table.to_csv(out_dir / "cells.csv", index=False)
    fields_table.to_csv(out_dir / "fields.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    classes = (("unipolar", "non_unipolar", "no_fluorescence") if args.rule == "half"
               else ("unipolar", "bipolar", "diffuse", "no_fluorescence"))
    viz.save_composition_bar(out_dir / "figures" / "composition.png", summary, classes=classes,
                             title="Localization pattern per condition")
    viz.save_unipolar_fraction(out_dir / "figures" / "unipolar_fraction.png", summary)
    viz.save_expression_box(out_dir / "figures" / "expression_level.png", cell_rows)
    if shift_grids:
        viz.save_shift_grid(out_dir / "figures" / "alignment_shift_grid.png", shift_grids)

    json.dump(
        dict(command=" ".join(sys.argv), pixel_size_um=args.pixel_size_um, rule=args.rule,
             segmentation=asdict(seg_params) if not mask_lookup else "external masks",
             alignment=asdict(align_params) if not args.no_align else "disabled",
             classification=asdict(polarity_params)),
        open(out_dir / "parameters.json", "w"), indent=2, default=str,
    )

    print("\n" + summary.to_string(index=False))
    print(f"\nwritten to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
