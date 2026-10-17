"""Choosing which parts of a large field to look at closely.

A 2048 x 2048 field holds hundreds of cells and only some of them express the
construct. For a figure, and for counting by eye, we want the few square
regions that contain the most cells with fluorescence actually inside them.
`find_disjoint_crops` slides a window over the field, scores each position by
that fraction, and returns the best non-overlapping regions, so the crops shown
side by side are genuinely different parts of the sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import regionprops


@dataclass
class CropCandidate:
    """One selected square region of the field."""

    rank: int
    top: int
    left: int
    size: int
    confinement_ratio: float    # fraction of cells in the crop with signal inside them
    cells_total: int
    cells_any_channel: int
    cells_all_channels: int     # cells where every channel has signal (co-labelled)
    cells_per_channel: dict[str, int]


def evaluate_confinement(labels: np.ndarray, channels: dict[str, np.ndarray],
                         sigma_factor: float = 3.5) -> tuple[dict[int, dict], dict]:
    """For every cell, record which channels have signal inside it.

    The noise floor of each channel is measured outside the masks, so a channel
    that is dim overall is not judged against a brighter one.
    """
    background = labels == 0
    floors = {}
    for name, image in channels.items():
        pixels = image[background] if background.any() else image
        median, std = float(np.median(pixels)), float(np.std(pixels))
        floors[name] = median + sigma_factor * std

    cells: dict[int, dict] = {}
    for prop in regionprops(labels):
        mask = labels == prop.label
        entry = {"label": prop.label, "centroid_y": prop.centroid[0],
                 "centroid_x": prop.centroid[1], "area": prop.area}
        confined = {}
        for name, image in channels.items():
            pixels = image[mask]
            peak = float(pixels.max()) if pixels.size else 0.0
            entry[f"max_{name}"] = peak
            confined[name] = bool(peak >= floors[name])
        entry["confined"] = confined
        entry["any_confined"] = any(confined.values())
        entry["all_confined"] = all(confined.values()) and bool(confined)
        cells[prop.label] = entry
    return cells, floors


def find_disjoint_crops(cells: dict[int, dict], image_shape: tuple[int, int],
                        channel_names: list[str], size: int = 512, stride: int = 32,
                        top_k: int = 3, min_cells: int = 10) -> list[CropCandidate]:
    """Return the best `top_k` square crops that do not overlap at all.

    Candidates are ranked by the fraction of cells with signal inside them,
    then by how many cells carry every channel, then by the number of labelled
    cells. Overlap between selected crops is exactly 0 px^2: a candidate that
    touches an already selected one is skipped rather than trimmed. If the
    field is too sparse to supply `top_k` crops, corner positions are added so
    that downstream code always receives the number of crops it expects.
    """
    height, width = image_shape
    candidates = []
    values = list(cells.values())

    for top in range(0, max(height - size + 1, 1), stride):
        for left in range(0, max(width - size + 1, 1), stride):
            inside = [c for c in values
                      if top <= c["centroid_y"] < top + size and left <= c["centroid_x"] < left + size]
            if len(inside) < min_cells:
                continue
            candidates.append(_score(inside, top, left, size, channel_names))

    candidates.sort(key=lambda c: (c.confinement_ratio, c.cells_all_channels, c.cells_any_channel),
                    reverse=True)

    selected: list[CropCandidate] = []
    for candidate in candidates:
        if _overlaps(candidate.top, candidate.left, size, selected):
            continue
        candidate.rank = len(selected) + 1
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    # The field may be too sparse, or the best crops may sit in the middle and
    # block every corner. Fall back to any free position, densest first, so the
    # caller still receives `top_k` crops where the image allows it.
    if len(selected) < top_k:
        fallback_positions = [(top, left)
                              for top in range(0, max(height - size + 1, 1), size // 2 or 1)
                              for left in range(0, max(width - size + 1, 1), size // 2 or 1)]
        fallback_positions += [(0, 0), (max(height - size, 0), 0),
                               (max(height - size, 0), max(width - size, 0)),
                               (0, max(width - size, 0))]
        scored_fallbacks = []
        for top, left in dict.fromkeys(fallback_positions):
            inside = [c for c in values
                      if top <= c["centroid_y"] < top + size and left <= c["centroid_x"] < left + size]
            scored_fallbacks.append(_score(inside, top, left, size, channel_names))
        scored_fallbacks.sort(key=lambda c: (c.confinement_ratio, c.cells_total), reverse=True)
        for candidate in scored_fallbacks:
            if len(selected) >= top_k:
                break
            if _overlaps(candidate.top, candidate.left, size, selected):
                continue
            candidate.rank = len(selected) + 1
            selected.append(candidate)
    return selected


def _score(inside: list[dict], top: int, left: int, size: int,
           channel_names: list[str]) -> CropCandidate:
    total = len(inside)
    any_confined = sum(1 for c in inside if c["any_confined"])
    all_confined = sum(1 for c in inside if c["all_confined"])
    per_channel = {name: sum(1 for c in inside if c["confined"].get(name)) for name in channel_names}
    return CropCandidate(
        rank=0, top=int(top), left=int(left), size=int(size),
        confinement_ratio=any_confined / total if total else 0.0,
        cells_total=total, cells_any_channel=any_confined,
        cells_all_channels=all_confined, cells_per_channel=per_channel,
    )


def _overlaps(top: int, left: int, size: int, selected: list[CropCandidate]) -> bool:
    """True if this position shares any pixel with an already selected crop."""
    for other in selected:
        overlap_h = max(0, min(top + size, other.top + other.size) - max(top, other.top))
        overlap_w = max(0, min(left + size, other.left + other.size) - max(left, other.left))
        if overlap_h * overlap_w > 0:
            return True
    return False
