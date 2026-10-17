#!/usr/bin/env python3
"""End-to-end check on synthetic data: run it after changing anything.

The demo images are built with a known shift between the phase and the
fluorescence channel and a known fraction of unipolar cells, so the test can
assert that the pipeline recovers both. It needs no microscope data and no
Omnipose: ground-truth masks are supplied with --masks.

    python tests/test_smoke.py          # plain run, prints PASS/FAIL
    pytest tests/test_smoke.py          # same checks under pytest
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Shifts applied by make_demo_data.py, which the alignment has to find again.
EXPECTED_SHIFTS = {"low": (7, -4), "mid": (-11, 9), "high": (3, 14)}


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"command failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}")


def test_pipeline(tmp_path: Path | None = None) -> None:
    """Build demo data, run the polarity and adhesion pipelines, check the results."""
    directory = Path(tmp_path or tempfile.mkdtemp())
    data = directory / "demo_data"
    _run([sys.executable, str(ROOT / "scripts" / "make_demo_data.py"), "--out", str(data)])

    polarity_out = directory / "polarity"
    command = [sys.executable, str(ROOT / "scripts" / "analyze_polarity.py"),
               "--tile", "256", "--out", str(polarity_out)]
    for name in EXPECTED_SHIFTS:
        command += ["--field", name, str(data / f"{name}_phase.tif"), str(data / f"{name}_fluor.tif")]
    for name in EXPECTED_SHIFTS:
        command += ["--masks", name, str(data / f"{name}_masks.npy")]
    _run(command)

    fields = pd.read_csv(polarity_out / "fields.csv")
    for _, row in fields.iterrows():
        expected = EXPECTED_SHIFTS[row.condition]
        found = (row.global_dy, row.global_dx)
        assert abs(found[0] - expected[0]) <= 2 and abs(found[1] - expected[1]) <= 2, (
            f"{row.condition}: expected a shift near {expected}, got {found}")
        assert row.inclusion_tile > row.inclusion_raw, (
            f"{row.condition}: the correction did not increase the in-mask signal")

    summary = pd.read_csv(polarity_out / "summary.csv").set_index("condition")
    unipolar = summary.unipolar_among_fluorescent_pct
    assert unipolar["low"] > unipolar["mid"] > unipolar["high"], (
        f"the unipolar fraction should fall with induction, got {unipolar.to_dict()}")
    assert unipolar["low"] > 70, f"expected a mostly unipolar low condition, got {unipolar['low']:.0f}%"

    for name in EXPECTED_SHIFTS:
        assert list((polarity_out / "figures").glob(f"{name}_*_classification.png")), (
            f"no classification figure for {name}")

    adhesion_out = directory / "adhesion"
    _run([sys.executable, str(ROOT / "scripts" / "analyze_adhesion.py"),
          "--field", "mixed", str(data / "mixed_phase.tif"),
          str(data / "mixed_red.tif"), str(data / "mixed_green.tif"),
          "--masks", "mixed", str(data / "mixed_masks.npy"),
          "--channel-names", "red_strain", "green_strain",
          "--crop-size", "256", "--top-k", "1", "--min-cells", "5",
          "--out", str(adhesion_out)])
    crops = pd.read_csv(adhesion_out / "crops.csv")
    assert len(crops) >= 1
    assert (crops.in_mask_after_red_strain > crops.in_mask_before_red_strain).all(), (
        "aligning the red channel did not move signal into the cells")

    benchmark_out = directory / "benchmark"
    _run([sys.executable, str(ROOT / "scripts" / "benchmark_segmentation.py"),
          "--phase", str(data / "low_phase.tif"), "--skip-omnipose",
          "--crop-size", "200", "--n-auto-crops", "1", "--out", str(benchmark_out)])
    stats = pd.read_csv(benchmark_out / "segmentation_stats.csv")
    assert set(stats.method) == {"otsu", "sauvola", "texture"}
    assert (stats.n_objects > 10).all(), "a classical method found almost nothing on clean demo data"


def test_units() -> None:
    """A few unit-level invariants that are easy to break."""
    from ecoli_imaging.polarity import wilson_interval
    from ecoli_imaging.utils import shift_image

    low, high = wilson_interval(9, 10)
    assert 0 <= low < 90 < high <= 100

    image = np.zeros((20, 20), dtype=np.float32)
    image[10, 10] = 1.0
    moved = shift_image(image, 3, -2, fill=0.0)
    assert moved[13, 8] == 1.0, "shift_image moved the pixel the wrong way"
    assert moved.sum() == 1.0, "shift_image must not duplicate or wrap pixels"


if __name__ == "__main__":
    test_units()
    print("unit checks passed")
    with tempfile.TemporaryDirectory() as tmp:
        test_pipeline(Path(tmp))
    print("pipeline checks passed")
    print("PASS")
