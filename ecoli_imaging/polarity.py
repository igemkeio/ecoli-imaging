"""Deciding, cell by cell, whether the fluorescence sits at one pole.

Two rules are available.

`half` (used for the PopZ dose-response in our wiki): a cell is unipolar when
at least 90 % of its focus signal lies in one half of the long axis. The centre
of an E. coli cell is occupied by the nucleoid, so a polar protein accumulates
towards the poles; when the cluster grows it pushes the nucleoid aside and the
signal extends from one pole towards mid-cell. That is still one-pole
localization, and a rule that only looks at the outermost 25 % of the cell
discards exactly those cells. Asking whether the signal stays within one half
keeps them.

`profile` (used for the qualitative two-colour work): bins the intensity along
the long axis and compares the two ends with mid-cell, giving a four-way call
of unipolar / bipolar / diffuse / no_signal. It can separate bipolar from
diffuse cells, but note that in evenly filled cells small fluctuations at the
two ends are easily read as two poles, so bipolar counts from this rule should
be checked by eye.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .utils import DEFAULT_PIXEL_SIZE_UM, robust_stats

HALF_CLASSES = ("no_fluor", "unipolar", "non_unipolar", "ambiguous")
PROFILE_CLASSES = ("no_signal", "unipolar", "bipolar", "diffuse")


@dataclass
class PolarityParams:
    """Thresholds for the per-cell call."""

    ring_inner_px: int = 3        # local background ring starts this far outside the cell
    ring_outer_px: int = 10       # ... and ends here
    signal_k: float = 3.0         # a pixel counts as signal above local background + k * sigma
    min_signal_px: int = 3        # fewer signal pixels than this -> no fluorescence
    focus_level: float = 0.4      # focus pixels are those at >= 40 % of the cell's peak excess
    half_min: float = 0.90        # `half` rule: unipolar if this much focus signal is in one half
    spill_ring_px: int = 3        # ring just outside the cell used for the spill-over check
    spill_ratio: float = 1.0      # peak outside > ratio * peak inside -> ambiguous, excluded
    pole_frac: float = 0.25       # `profile` rule: each pole is the terminal 25 % of the long axis
    profile_bins: int = 20
    polarity_threshold: float = 1.35   # `profile` rule: pole must exceed mid-cell by this factor
    bipolar_balance: float = 0.55      # weaker pole / stronger pole above this -> bipolar


def long_axis_position(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position of every pixel along the cell's principal axis, scaled to 0-1.

    Returns (positions, axis_vector, centroid). The range is padded by half a
    pixel at each end so that terminal pixels do not land exactly on 0 or 1.
    """
    points = coords.astype(float)
    centroid = points.mean(axis=0)
    centred = points - centroid
    covariance = np.cov(centred.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection = centred @ axis
    low, high = projection.min() - 0.5, projection.max() + 0.5
    span = max(high - low, 1e-9)
    return (projection - low) / span, axis, centroid


def classify_cells(
    fluor: np.ndarray,
    labels: np.ndarray,
    params: PolarityParams | None = None,
    rule: str = "half",
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    with_profiles: bool = False,
) -> list[dict]:
    """Classify every cell in a field.

    `fluor` must already be aligned to `labels`; pass the expanded masks from
    `segmentation.measurement_masks` so a focus sitting on the membrane is not
    cut off. One dictionary is returned per cell.
    """
    params = params or PolarityParams()
    if rule not in ("half", "profile"):
        raise ValueError("rule must be 'half' or 'profile'")

    all_cells = labels > 0
    dilated_all = ndi.binary_dilation(all_cells, iterations=params.ring_inner_px)
    far_background = ~ndi.binary_dilation(all_cells, iterations=6)
    global_median, global_sigma = robust_stats(fluor[far_background][::7])

    rows: list[dict] = []
    objects = ndi.find_objects(labels)
    height, width = labels.shape
    pad = params.ring_outer_px + 2

    for value, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, height)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, width)
        window_labels = labels[y0:y1, x0:x1]
        mask = window_labels == value
        window = fluor[y0:y1, x0:x1].astype(float)

        # Local background from a ring around the cell that excludes neighbours.
        outer = ndi.binary_dilation(mask, iterations=params.ring_outer_px)
        ring = outer & ~dilated_all[y0:y1, x0:x1]
        if ring.sum() >= 30:
            bg_median, bg_sigma = robust_stats(window[ring])
        else:
            bg_median, bg_sigma = global_median, global_sigma

        values = window[mask]
        coords = np.argwhere(mask)
        positions, axis, centroid = long_axis_position(coords)
        signal = values > bg_median + params.signal_k * bg_sigma
        excess = np.clip(values - bg_median, 0, None)
        length_um = float(np.ptp(coords @ axis) * pixel_size_um) if len(coords) > 1 else 0.0

        row = dict(
            label=value,
            centroid_y=float(centroid[0] + y0),
            centroid_x=float(centroid[1] + x0),
            area_um2=float(mask.sum() * pixel_size_um ** 2),
            length_um=length_um,
            background_median=float(bg_median),
            background_sigma=float(bg_sigma),
            mean_intensity=float(values.mean()),
            mean_excess=float(excess.mean()),
            total_excess=float(excess.sum()),
            max_intensity=float(values.max()),
            n_signal_px=int(signal.sum()),
        )

        # Spill-over check: if the brightest pixel just outside the cell beats
        # anything inside, the 'signal' is the halo of a neighbouring spot.
        near = ndi.binary_dilation(mask, iterations=params.spill_ring_px) & ~mask
        peak_outside = float((window[near] - bg_median).max()) if near.any() else 0.0
        peak_inside = float(excess.max())
        row.update(peak_inside=peak_inside, peak_outside=peak_outside)

        if signal.sum() < params.min_signal_px:
            row.update(_empty_call(rule, "no_fluor" if rule == "half" else "no_signal"))
        elif rule == "half" and peak_outside > params.spill_ratio * peak_inside:
            row.update(_empty_call(rule, "ambiguous"))
        else:
            # Focus pixels: significant AND bright relative to this cell's peak.
            # Using only the bright core stops the optical blur around a polar
            # spot from leaking into mid-cell and looking like diffuse signal.
            focus = signal & (excess >= params.focus_level * excess.max())
            weighted = excess * focus
            total = float(weighted.sum())
            if total <= 0:
                row.update(_empty_call(rule, "no_fluor" if rule == "half" else "no_signal"))
            elif rule == "half":
                first_half = float(weighted[positions < 0.5].sum()) / total
                half_frac = max(first_half, 1.0 - first_half)
                pole_a = float(weighted[positions < params.pole_frac].sum()) / total
                pole_b = float(weighted[positions > 1 - params.pole_frac].sum()) / total
                row.update(
                    half_frac=half_frac,
                    pole_a_frac=pole_a,
                    pole_b_frac=pole_b,
                    mid_frac=1.0 - pole_a - pole_b,
                    classification="unipolar" if half_frac >= params.half_min else "non_unipolar",
                )
            else:
                row.update(_profile_call(values, positions, bg_median, params))

        if with_profiles:
            bins = params.profile_bins
            index = np.minimum((positions * bins).astype(int), bins - 1)
            profile = np.array([
                values[index == b].mean() - bg_median if (index == b).any() else np.nan
                for b in range(bins)
            ])
            row["profile"] = profile
        rows.append(row)
    return rows


def _empty_call(rule: str, classification: str) -> dict:
    if rule == "half":
        return dict(half_frac=np.nan, pole_a_frac=np.nan, pole_b_frac=np.nan,
                    mid_frac=np.nan, classification=classification)
    return dict(pole_max=np.nan, pole_min=np.nan, mid_mean=np.nan,
                polar_ratio=np.nan, classification=classification)


def _profile_call(values: np.ndarray, positions: np.ndarray, bg_median: float,
                  params: PolarityParams) -> dict:
    """Four-way call from the binned long-axis profile (unipolar / bipolar / diffuse)."""
    bins = params.profile_bins
    index = np.minimum((positions * bins).astype(int), bins - 1)
    profile = np.zeros(bins, dtype=float)
    counts = np.zeros(bins, dtype=float)
    np.add.at(profile, index, values)
    np.add.at(counts, index, 1.0)
    filled = counts > 0
    profile[filled] /= counts[filled]
    profile[~filled] = bg_median

    edge = max(bins // 7, 1)           # ~15 % of the cell at each end
    middle = slice(bins // 4, bins - bins // 4)
    end_a = float(profile[:edge].mean())
    end_b = float(profile[-edge:].mean())
    mid_mean = float(profile[middle].mean()) if filled[middle].any() else 1e-3
    pole_max, pole_min = max(end_a, end_b), min(end_a, end_b)
    polar_ratio = pole_max / max(mid_mean, 1e-3)

    strong_pole = pole_max > mid_mean * params.polarity_threshold
    weak_pole = pole_min > mid_mean * params.polarity_threshold
    if strong_pole and weak_pole and pole_min >= pole_max * params.bipolar_balance:
        classification = "bipolar"
    elif strong_pole:
        classification = "unipolar"
    else:
        classification = "diffuse"
    return dict(pole_max=pole_max, pole_min=pole_min, mid_mean=mid_mean,
                polar_ratio=polar_ratio, classification=classification)


def summarise(cells, group_key: str = "condition", only_strict: bool = True):
    """Per-condition counts and percentages from a cell table.

    Percentages are given both over all classified cells and over fluorescent
    cells only, since the second is what a dose-response curve should show.
    """
    import pandas as pd

    table = pd.DataFrame(cells)
    if only_strict and "strict_ok" in table.columns:
        table = table[table.strict_ok]
    table = table[table.classification != "ambiguous"]

    rows = []
    for key, group in table.groupby(group_key, sort=False):
        classified = len(group)
        counts = group.classification.value_counts()
        no_signal = int(counts.get("no_fluor", 0) + counts.get("no_signal", 0))
        unipolar = int(counts.get("unipolar", 0))
        fluorescent = classified - no_signal
        row = {
            group_key: key,
            "cells_classified": classified,
            "cells_fluorescent": fluorescent,
            "no_fluorescence_pct": 100.0 * no_signal / classified if classified else np.nan,
            "unipolar_pct": 100.0 * unipolar / classified if classified else np.nan,
            "unipolar_among_fluorescent_pct": 100.0 * unipolar / fluorescent if fluorescent else np.nan,
        }
        for name in ("non_unipolar", "bipolar", "diffuse"):
            if name in counts:
                row[f"{name}_pct"] = 100.0 * int(counts[name]) / classified
        low, high = wilson_interval(unipolar, fluorescent)
        row["unipolar_among_fluorescent_lo"] = low
        row["unipolar_among_fluorescent_hi"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % confidence interval for a proportion, in percent.

    Preferred over the normal approximation because the counts per condition
    are small and the proportions sit close to 0 or 1.
    """
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominator
    return 100.0 * max(centre - margin, 0.0), 100.0 * min(centre + margin, 1.0)
