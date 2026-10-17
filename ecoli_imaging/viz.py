"""Figures: QC panels that show what the pipeline did, and result plots.

Everything is drawn with matplotlib's Agg backend so the scripts run on a
headless machine. Colours are consistent across the figures and were chosen to
stay distinguishable in common forms of colour blindness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries

from .utils import DEFAULT_PIXEL_SIZE_UM, normalize

CLASS_COLORS = {
    "unipolar": "#1f77b4",
    "non_unipolar": "#ff7f0e",
    "bipolar": "#e6a000",
    "diffuse": "#9467bd",
    "no_fluor": "#b0bec5",
    "no_signal": "#b0bec5",
    "ambiguous": "#f2c200",
}

CHANNEL_RGB = {
    "red": (1.0, 0.25, 0.25),
    "green": (0.25, 1.0, 0.35),
    "magenta": (1.0, 0.35, 1.0),
    "cyan": (0.35, 1.0, 1.0),
    "yellow": (1.0, 0.95, 0.3),
}

RANK_COLORS = {1: "#ff1744", 2: "#ffd600", 3: "#00e5ff"}


def add_scale_bar(ax, length_um: float, pixel_size_um: float, color: str = "white") -> None:
    """Draw a scale bar in the lower-right corner of an image axis."""
    length_px = length_um / pixel_size_um
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()[1], ax.get_ylim()[0]
    width = abs(x1 - x0)
    height = abs(y0 - y1)
    x = x0 + 0.95 * width - length_px
    y = y1 + 0.92 * height if y1 < y0 else y0 + 0.92 * height
    ax.add_patch(Rectangle((x, y), length_px, max(height * 0.012, 2), color=color, zorder=10))
    ax.text(x + length_px / 2, y - height * 0.02, f"{length_um:g} um",
            color=color, ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=10)


def merge_rgb(phase: np.ndarray, channels: dict[str, tuple[np.ndarray, str]],
              phase_weight: float = 0.75) -> np.ndarray:
    """Compose a display image: grayscale phase plus one colour per channel.

    `channels` maps a channel name to (image, colour name). Channels are added,
    so a cell carrying two of them appears in the mixed colour.
    """
    base = normalize(phase) * phase_weight
    rgb = np.dstack([base, base, base])
    for _, (image, color) in channels.items():
        tint = np.array(CHANNEL_RGB.get(color, (1.0, 1.0, 1.0)))
        # Stretch each channel from its own background level, so camera offset
        # and read noise do not tint the whole frame.
        image = np.asarray(image, dtype=np.float32)
        low = float(np.median(image))
        high = float(np.percentile(image, 99.7))
        layer = np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
        rgb += layer[..., None] * tint[None, None, :]
    return np.clip(rgb, 0, 1)


def save_merged(path: str | Path, phase: np.ndarray, channels: dict[str, tuple[np.ndarray, str]],
                labels: np.ndarray | None = None, pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
                scale_bar_um: float = 5.0, dpi: int = 200) -> None:
    """Merged image on its own, with cell outlines and a scale bar."""
    rgb = merge_rgb(phase, channels)
    if labels is not None:
        outline = find_boundaries(labels, mode="outer")
        rgb[outline] = 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_axis_off()
    add_scale_bar(ax, scale_bar_um, pixel_size_um)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_alignment_comparison(path: str | Path, phase: np.ndarray,
                              before: dict[str, tuple[np.ndarray, str]],
                              after: dict[str, tuple[np.ndarray, str]],
                              labels: np.ndarray, shifts: dict[str, tuple[int, int]],
                              scores: dict[str, tuple[float, float]] | None = None,
                              pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
                              title: str = "", dpi: int = 160) -> None:
    """Side-by-side 'as acquired' versus 'after shift correction'.

    This is the figure that shows the correction was needed at all: before it,
    the fluorescence sits beside the cell outlines rather than inside them.
    """
    outline = find_boundaries(labels, mode="outer")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    for ax, data, label in ((axes[0], before, "As acquired"), (axes[1], after, "After shift correction")):
        rgb = merge_rgb(phase, data)
        rgb[outline] = 1.0
        ax.imshow(rgb)
        ax.set_axis_off()
        ax.set_title(label, fontsize=13)
        add_scale_bar(ax, 5.0, pixel_size_um)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if shifts:
        # The shift values go under the panels, where they cannot collide with
        # the panel titles however many channels there are.
        detail = "\n".join(
            f"{name}: dy = {dy:+d} px, dx = {dx:+d} px"
            + (f"   ({scores[name][0]*100:.1f}% -> {scores[name][1]*100:.1f}% of the signal inside cells)"
               if scores and name in scores else "")
            for name, (dy, dx) in shifts.items()
        )
        fig.subplots_adjust(bottom=0.09 + 0.035 * len(shifts))
        fig.text(0.5, 0.01, detail, ha="center", va="bottom", fontsize=11)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_classification_overlay(path: str | Path, phase: np.ndarray, fluor: np.ndarray,
                                labels: np.ndarray, cells: list[dict],
                                pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
                                title: str = "", color: str = "magenta", dpi: int = 160) -> None:
    """Cell outlines coloured by the call the pipeline made.

    This is the figure to look at before trusting any number: wrong calls are
    usually obvious here, and the legend doubles as the class list.
    """
    rgb = merge_rgb(phase, {"fluor": (fluor, color)})
    classification = {c["label"]: c.get("classification", "ambiguous") for c in cells}
    strict = {c["label"]: c.get("strict_ok", True) for c in cells}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    for value, name in classification.items():
        outline = find_boundaries(labels == value, mode="outer")
        ys, xs = np.nonzero(outline)
        if ys.size == 0:
            continue
        colour = CLASS_COLORS.get(name, "#ffffff")
        ax.scatter(xs, ys, s=0.6, c=colour, marker="s", linewidths=0,
                   alpha=0.45 if not strict.get(value, True) else 1.0)
    ax.set_axis_off()
    add_scale_bar(ax, 5.0, pixel_size_um)
    if title:
        ax.set_title(title, fontsize=13)
    present = [n for n in ("unipolar", "non_unipolar", "bipolar", "diffuse", "no_fluor",
                           "no_signal", "ambiguous") if n in set(classification.values())]
    handles = [plt.Line2D([], [], color=CLASS_COLORS[n], lw=4, label=n.replace("_", " ")) for n in present]
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_overview(path: str | Path, phase: np.ndarray, crops, title: str = "",
                  pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM, dpi: int = 140) -> None:
    """Whole field with the selected crops outlined, so a reader can see where they came from."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normalize(phase), cmap="gray")
    for crop in crops:
        colour = RANK_COLORS.get(crop.rank, "#ffffff")
        ax.add_patch(Rectangle((crop.left, crop.top), crop.size, crop.size,
                               fill=False, edgecolor=colour, linewidth=2.2))
        ax.text(crop.left + 8, crop.top + 28, f"Rank {crop.rank}  {crop.confinement_ratio*100:.0f}%",
                color=colour, fontsize=11, fontweight="bold")
    ax.set_axis_off()
    add_scale_bar(ax, 20.0, pixel_size_um)
    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_qc_panel(path: str | Path, phase: np.ndarray, labels: np.ndarray,
                  channels: dict[str, tuple[np.ndarray, str]], cells: list[dict],
                  title: str = "", pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
                  dpi: int = 150) -> None:
    """Four panels for one crop: phase + masks, fluorescence, merge, calls."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12.6))
    outline = find_boundaries(labels, mode="outer")

    base = normalize(phase)
    rgb = np.dstack([base, base, base])
    rgb[outline] = (0.4, 1.0, 1.0)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"Phase contrast + cell masks (n = {int(labels.max())})", fontsize=12)

    fluor_only = merge_rgb(np.zeros_like(phase), channels, phase_weight=0.0)
    axes[0, 1].imshow(fluor_only)
    axes[0, 1].set_title("Fluorescence " + ", ".join(f"{n} ({c})" for n, (_, c) in channels.items()), fontsize=12)

    merged = merge_rgb(phase, channels)
    merged[outline] = 1.0
    axes[1, 0].imshow(merged)
    axes[1, 0].set_title("Merge after shift correction", fontsize=12)

    call_rgb = np.dstack([base * 0.5] * 3)
    counts: dict[str, int] = {}
    for cell in cells:
        name = cell.get("classification", "ambiguous")
        counts[name] = counts.get(name, 0) + 1
        mask = labels == cell["label"]
        colour = np.array(matplotlib.colors.to_rgb(CLASS_COLORS.get(name, "#ffffff")))
        call_rgb[mask] = colour
    axes[1, 1].imshow(call_rgb)
    axes[1, 1].set_title("Per-cell calls: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), fontsize=11)

    for ax in axes.ravel():
        ax.set_axis_off()
        add_scale_bar(ax, 5.0, pixel_size_um)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_shift_grid(path: str | Path, grids: dict[str, np.ndarray], dpi: int = 150) -> None:
    """Quiver plot of the per-tile shift for every field.

    Worth including in a report: it shows at a glance that the offset is not
    one constant number but varies between fields and within a field.
    """
    names = list(grids)
    columns = min(len(names), 5)
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        grid = grids[name]
        ny, nx = grid.shape[:2]
        ys, xs = np.mgrid[0:ny, 0:nx]
        ax.quiver(xs, -ys, grid[..., 1], -grid[..., 0], angles="xy", color="#1f77b4")
        for i in range(ny):
            for j in range(nx):
                ax.text(j, -i - 0.32, f"{grid[i, j, 0]},{grid[i, j, 1]}", ha="center", fontsize=7, color="#444")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(names):]:
        ax.set_axis_off()
    fig.suptitle("Per-tile shift of the fluorescence image relative to phase contrast (dy, dx in px)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_composition_bar(path: str | Path, summary, group_key: str = "condition",
                         classes: tuple[str, ...] = ("unipolar", "non_unipolar", "no_fluorescence"),
                         dpi: int = 160, title: str = "") -> None:
    """Stacked bars: what fraction of the cells fell into each class per condition."""
    fig, ax = plt.subplots(figsize=(1.6 * max(len(summary), 3) + 2.5, 5.2))
    bottom = np.zeros(len(summary))
    x = np.arange(len(summary))
    for name in classes:
        column = f"{name}_pct"
        if column not in summary.columns:
            continue
        values = summary[column].to_numpy(dtype=float)
        values = np.nan_to_num(values)
        colour = CLASS_COLORS.get(name, "#b0bec5")
        ax.bar(x, values, bottom=bottom, color=colour, label=name.replace("_", " "), width=0.62)
        for xi, (v, b) in enumerate(zip(values, bottom)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=11)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row[group_key]}\nn={int(row.cells_classified)}" for _, row in summary.iterrows()])
    ax.set_ylabel("Cells (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title or "Localization pattern per condition")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_unipolar_fraction(path: str | Path, summary, group_key: str = "condition",
                           dpi: int = 160, title: str = "") -> None:
    """Unipolar fraction among fluorescent cells, with Wilson confidence intervals."""
    x = np.arange(len(summary))
    y = summary.unipolar_among_fluorescent_pct.to_numpy(dtype=float)
    lo = summary.unipolar_among_fluorescent_lo.to_numpy(dtype=float)
    hi = summary.unipolar_among_fluorescent_hi.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.5 * max(len(summary), 3) + 2.0, 5.0))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", color=CLASS_COLORS["unipolar"],
                capsize=5, markersize=8, linewidth=2)
    for xi, (yi, n) in enumerate(zip(y, summary.cells_fluorescent)):
        ax.annotate(f"{yi:.0f}%\nn={int(n)}", (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(summary[group_key].astype(str))
    ax.set_ylabel("Unipolar among fluorescent cells (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title or "Unipolar fraction (95 % CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_expression_box(path: str | Path, cells, group_key: str = "condition",
                        dpi: int = 160, title: str = "") -> None:
    """Per-cell background-subtracted signal, one box per condition (log scale).

    Reading this next to the classification plot tells you whether a condition
    behaved oddly because of biology or simply because nothing was expressed.
    """
    import pandas as pd

    table = pd.DataFrame(cells)
    if "strict_ok" in table.columns:
        table = table[table.strict_ok]
    groups = [g for _, g in table.groupby(group_key, sort=False)]
    names = [str(k) for k, _ in table.groupby(group_key, sort=False)]
    data = [np.clip(g.mean_excess.to_numpy(dtype=float), 1e-2, None) for g in groups]

    fig, ax = plt.subplots(figsize=(1.5 * max(len(names), 3) + 2.0, 5.0))
    # Tick labels are set separately because the boxplot keyword for them was
    # renamed in matplotlib 3.9; this works on either version.
    ax.boxplot(data, showfliers=False,
               boxprops=dict(color="#1f77b4"), medianprops=dict(color="black", linewidth=2))
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names)
    for i, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.05, size=len(values))
        ax.plot(i + jitter, values, ".", color="#1f77b4", alpha=0.35, markersize=4)
    ax.set_yscale("log")
    ax.set_ylabel("Mean signal per cell above background (a.u.)")
    ax.set_title(title or "Expression level per cell")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
