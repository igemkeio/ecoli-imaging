"""Cell segmentation with Omnipose, plus the mask clean-up that follows it.

Why Omnipose: classical thresholding (Otsu, Sauvola, local texture - see
`classical.py`) segments phase-contrast images of E. coli well only where the
field is in focus. Where cells are surrounded by interference fringes, those
methods mark background shading as cells, leave cell interiors hollow or
inflate outlines out to the halo. The `bact_phase_omni` model, trained on
rod-shaped bacteria, still resolves individual cells there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import relabel_sequential
from skimage.transform import resize

from .utils import DEFAULT_PIXEL_SIZE_UM

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class MaskCleanupParams:
    """Size windows and clean-up settings, all in micrometres.

    Two windows are used, exactly as in the published analysis:

    `len_um` / `wid_um` / `area_um2` define what counts as a *cell* at all.
    Every object inside this window takes part in alignment, local background
    estimation and neighbour competition, so it is deliberately generous.

    `strict_*` is the narrower window that selects the cells actually
    *analysed*: normally grown, non-dividing cells. Objects outside it stay in
    the mask image (otherwise their fluorescence would leak into a neighbour's
    background ring) but are flagged with strict_ok = False.
    """

    len_um: tuple[float, float] = (1.0, 8.0)
    wid_um: tuple[float, float] = (0.55, 1.6)
    area_um2: tuple[float, float] = (0.4, 6.0)
    strict_len_um: tuple[float, float] = (2.0, 4.0)
    strict_wid_um: tuple[float, float] = (0.8, 1.2)
    strict_area_um2: tuple[float, float] = (1.4, 4.2)
    sliver_wid_um: float = 0.55       # fragments thinner than this are merged into a neighbour
    sliver_area_um2: float = 0.4
    merge_max_wid_um: float = 1.6     # a merge is accepted only if the union stays cell-shaped
    merge_min_solidity: float = 0.80
    border_px: int = 5                # drop cells touching the image border
    close_px: int = 3                 # radius used to fill hollow masks


@dataclass
class SegmentationParams:
    """Omnipose inference settings."""

    model_type: str = "bact_phase_omni"
    scale: float = 1.5          # upscale before inference: cells are only ~6 px wide at 0.1625 um/px
    mask_threshold: float = 0.0
    flow_threshold: float = 0.4
    use_gpu: bool = False
    polarity: str = "both"      # "inverted", "normal" or "both" (see segment_phase)
    cleanup: MaskCleanupParams = field(default_factory=MaskCleanupParams)


# --------------------------------------------------------------------------- inference
def _get_model(model_type: str, use_gpu: bool):
    """Load an Omnipose model once per process (loading takes several seconds)."""
    key = f"{model_type}:{use_gpu}"
    if key not in _MODEL_CACHE:
        try:
            from cellpose_omni import models
        except ImportError as exc:  # pragma: no cover - depends on the user's install
            raise ImportError(
                "Omnipose is required for segmentation. Install it with\n"
                "    pip install omnipose\n"
                "or pass pre-computed label images with --masks."
            ) from exc
        _MODEL_CACHE[key] = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    return _MODEL_CACHE[key]


def _normalize_for_model(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [0.5, 99.5])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _run_omnipose(image: np.ndarray, params: SegmentationParams, invert: bool) -> np.ndarray:
    """Run the model once on the phase image, optionally inverted."""
    model = _get_model(params.model_type, params.use_gpu)
    h, w = image.shape
    scaled = resize(image, (int(h * params.scale), int(w * params.scale)), order=1, preserve_range=True)
    normed = _normalize_for_model(scaled)
    if invert:
        normed = 1.0 - normed
    masks, _, _ = model.eval(
        normed,
        channels=None,
        rescale=None,
        mask_threshold=params.mask_threshold,
        flow_threshold=params.flow_threshold,
        omni=True,
        cluster=True,
        tile=True,
        verbose=False,
    )
    # Back to the original grid with nearest-neighbour so label values survive.
    masks = resize(masks.astype(np.int32), (h, w), order=0, preserve_range=True, anti_aliasing=False)
    return masks.astype(np.int32)


def segment_phase(
    phase: np.ndarray,
    params: SegmentationParams | None = None,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    enrichment_image: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Segment a phase-contrast field and return (clean_labels, stats).

    Slightly defocused cells appear bright-centred in phase contrast, and the
    model finds them in the *inverted* image; well-focused cells are found in
    the normal image. With polarity="both" the model is run twice and the two
    results are combined: the better run provides the cells, and isolated cells
    from the other run are added.

    `enrichment_image` (a fluorescence image of the same field) decides which
    run is better, by how strongly the fluorescence signal is concentrated
    inside its masks. Without it, the run with more cells wins.
    """
    params = params or SegmentationParams()
    stats: dict = {}
    variants: dict[str, np.ndarray] = {}

    if params.polarity in ("inverted", "both"):
        variants["inverted"] = _run_omnipose(phase, params, invert=True)
    if params.polarity in ("normal", "both"):
        variants["normal"] = _run_omnipose(phase, params, invert=False)
    if not variants:
        raise ValueError("polarity must be 'inverted', 'normal' or 'both'")

    cleaned = {name: clean_masks(raw, params.cleanup, pixel_size_um) for name, raw in variants.items()}

    if len(cleaned) == 1:
        name = next(iter(cleaned))
        labels, st = cleaned[name]
        stats.update(st)
        stats["segmentation_variant"] = name
        stats["cells_added_from_other_variant"] = 0
        return labels, stats

    scores = {}
    for name, (labels, _) in cleaned.items():
        if enrichment_image is not None:
            scores[name] = signal_enrichment(enrichment_image, labels)
        else:
            scores[name] = float(labels.max())
    primary = max(scores, key=scores.get)
    secondary = "normal" if primary == "inverted" else "inverted"
    labels, added = union_masks(cleaned[primary][0], cleaned[secondary][0])
    stats.update(cleaned[primary][1])
    stats["segmentation_variant"] = primary
    stats["cells_added_from_other_variant"] = added
    stats["variant_scores"] = scores
    return labels, stats


def signal_enrichment(fluor: np.ndarray, labels: np.ndarray, k: float = 5.0) -> float:
    """How concentrated the bright fluorescence pixels are inside the masks.

    Dividing by the mask area fraction keeps wide masks that merely cover phase
    halos from scoring well simply because they are large.
    """
    mask = labels > 0
    outside = ~ndi.binary_dilation(mask, iterations=6)
    from .utils import robust_stats

    med, sigma = robust_stats(fluor[outside][::7])
    bright = fluor > med + k * sigma
    if bright.sum() == 0:
        return 0.0
    inside_fraction = float((bright & mask).sum() / bright.sum())
    area_fraction = max(float(mask.mean()), 1e-9)
    return inside_fraction / area_fraction


# --------------------------------------------------------------------------- clean-up
def fill_labels(labels: np.ndarray, radius: int) -> np.ndarray:
    """Close and fill each label separately.

    On slightly defocused phase images the model often traces only the dark
    cell outline and leaves the bright centre empty, giving one label shaped
    like two parallel stripes. Closing with a disk and filling holes repairs
    this; only background pixels are claimed, so neighbours are never eaten.
    """
    out = labels.copy()
    structure = disk(radius)
    height, width = labels.shape
    for value, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - radius - 1, 0), min(sl[0].stop + radius + 1, height)
        x0, x1 = max(sl[1].start - radius - 1, 0), min(sl[1].stop + radius + 1, width)
        window = labels[y0:y1, x0:x1]
        mask = window == value
        closed = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=structure))
        free = closed & (out[y0:y1, x0:x1] == 0)
        out[y0:y1, x0:x1][free] = value
    # Keep only the largest connected component of each label.
    for value, sl in enumerate(ndi.find_objects(out), start=1):
        if sl is None:
            continue
        mask = out[sl] == value
        components, count = ndi.label(mask)
        if count > 1:
            sizes = ndi.sum(mask, components, range(1, count + 1))
            keep = components == (int(np.argmax(sizes)) + 1)
            out[sl][mask & ~keep] = 0
    return out


def clean_masks(
    raw: np.ndarray,
    params: MaskCleanupParams | None = None,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
) -> tuple[np.ndarray, dict]:
    """Repair, merge and size-filter raw model output.

    Steps: fill hollow masks -> merge sliver fragments into the neighbour they
    belong to -> drop objects outside the generous size window or touching the
    border. Returns (labels, stats).
    """
    params = params or MaskCleanupParams()
    labels = fill_labels(raw.astype(np.int32), params.close_px)
    stats = {"raw_labels": int(len(np.unique(labels)) - 1), "merged_slivers": 0}

    props = {r.label: r for r in regionprops(labels)}
    sliver_width_px = params.sliver_wid_um / pixel_size_um
    sliver_area_px = params.sliver_area_um2 / pixel_size_um ** 2
    slivers = [
        value for value, r in props.items()
        if r.axis_minor_length < sliver_width_px or r.area < sliver_area_px
    ]
    for value in slivers:
        if value not in props:
            continue
        r = props[value]
        y0, x0, y1, x1 = r.bbox
        y0, x0 = max(y0 - 2, 0), max(x0 - 2, 0)
        window = labels[y0:y1 + 2, x0:x1 + 2]
        mask = window == value
        ring = ndi.binary_dilation(mask, iterations=1) & ~mask
        neighbours, counts = np.unique(window[ring], return_counts=True)
        chosen = None
        for neighbour, _ in sorted(zip(neighbours, counts), key=lambda t: -t[1]):
            if neighbour in (0, value):
                continue
            union = (labels == neighbour) | (labels == value)
            union_props = regionprops(union.astype(np.uint8))[0]
            if (union_props.axis_minor_length * pixel_size_um <= params.merge_max_wid_um
                    and union_props.solidity >= params.merge_min_solidity):
                chosen = neighbour
                break
        if chosen is not None:
            labels[labels == value] = chosen
            props.pop(value)
            merged = regionprops((labels == chosen).astype(np.uint8))[0]
            merged.label = chosen
            props[chosen] = merged
            stats["merged_slivers"] += 1

    out = np.zeros_like(labels)
    height, width = labels.shape
    kept = 0
    stats.update(rejected_size=0, rejected_border=0)
    sizes = []
    for r in regionprops(labels):
        length = r.axis_major_length * pixel_size_um
        cell_width = r.axis_minor_length * pixel_size_um
        area = r.area * pixel_size_um ** 2
        y0, x0, y1, x1 = r.bbox
        b = params.border_px
        if y0 < b or x0 < b or y1 > height - b or x1 > width - b:
            stats["rejected_border"] += 1
            continue
        in_window = (
            params.len_um[0] <= length <= params.len_um[1]
            and params.wid_um[0] <= cell_width <= params.wid_um[1]
            and params.area_um2[0] <= area <= params.area_um2[1]
        )
        sizes.append((length, cell_width, area, float(in_window)))
        if not in_window:
            stats["rejected_size"] += 1
            continue
        kept += 1
        out[tuple(r.coords.T)] = kept
    stats["cells"] = kept
    stats["sizes"] = np.array(sizes) if sizes else np.zeros((0, 4))
    return out, stats


def union_masks(primary: np.ndarray, secondary: np.ndarray) -> tuple[np.ndarray, int]:
    """Add the cells of `secondary` that `primary` missed entirely.

    A secondary cell is added only when it neither overlaps nor comes within
    2 px of a primary cell, so a phase halo beside a real cell cannot become an
    extra 'cell'.
    """
    out = primary.copy()
    next_label = int(primary.max()) + 1
    added = 0
    for value, sl in enumerate(ndi.find_objects(secondary), start=1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - 3, 0), sl[0].stop + 3
        x0, x1 = max(sl[1].start - 3, 0), sl[1].stop + 3
        region = out[y0:y1, x0:x1]
        mask = secondary[y0:y1, x0:x1] == value
        near = ndi.binary_dilation(mask, iterations=2)
        if not (region[near] > 0).any():
            region[mask & (region == 0)] = next_label
            next_label += 1
            added += 1
    return relabel_sequential(out)[0], added


def size_table(labels: np.ndarray, params: MaskCleanupParams | None = None,
               pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM):
    """Per-cell size measurements and the strict selection flag.

    Length is the maximum Feret diameter (tip to tip), which matches a rod's
    real length better than the major axis of an equivalent ellipse.
    `strict_reason` records why a cell was excluded: short, long, width or area.
    """
    import pandas as pd

    params = params or MaskCleanupParams()
    rows = []
    for r in regionprops(labels):
        length = r.feret_diameter_max * pixel_size_um
        cell_width = r.axis_minor_length * pixel_size_um
        area = r.area * pixel_size_um ** 2
        if length < params.strict_len_um[0]:
            reason = "short"
        elif length > params.strict_len_um[1]:
            reason = "long"
        elif not (params.strict_wid_um[0] <= cell_width <= params.strict_wid_um[1]):
            reason = "width"
        elif not (params.strict_area_um2[0] <= area <= params.strict_area_um2[1]):
            reason = "area"
        else:
            reason = "ok"
        rows.append(
            dict(label=r.label, len_feret_um=length, wid_um=cell_width, area_um2=area,
                 strict_ok=reason == "ok", strict_reason=reason)
        )
    return pd.DataFrame(rows)


def measurement_masks(labels: np.ndarray, expand_px: int = 2) -> np.ndarray:
    """Expand every cell by a couple of pixels without letting neighbours overlap.

    A polar focus sits on the membrane at the very tip of the cell; optical
    blur (PSF FWHM ~0.4 um, i.e. ~2.5 px at 0.1625 um/px) plus 1-2 px of
    residual alignment error puts part of it just outside the phase outline.
    """
    from skimage.segmentation import expand_labels

    return expand_labels(labels, distance=expand_px)
