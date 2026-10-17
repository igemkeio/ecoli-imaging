"""Quantitative analysis of fluorescence localization and cell-cell adhesion in E. coli.

Written for iGEM Keio 2026 ("One Direction") and generalized so that any lab
with phase-contrast plus fluorescence images of rod-shaped bacteria can run it.

Three pipelines live in `scripts/`:

  analyze_polarity.py      how fluorescence is distributed along each cell, and
                           how that changes across an induction series
  analyze_adhesion.py      two-colour crops of mixed cultures, used to show
                           whether cells of different colours stick together
  benchmark_segmentation.py  classical thresholding against Omnipose on your
                           own images

The building blocks are importable on their own; see the README for the
shortest path from a pair of TIFFs to a result.
"""

__version__ = "1.0.0"

from . import alignment, classical, crops, imageio, polarity, segmentation, utils, viz  # noqa: F401

__all__ = [
    "alignment",
    "classical",
    "crops",
    "imageio",
    "polarity",
    "segmentation",
    "utils",
    "viz",
]
