"""Reading microscope images and label masks from whatever the acquisition software produced."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: str | Path, frame: int | None = None) -> np.ndarray:
    """Load a single 2-D grayscale image.

    Accepted inputs:
      * a TIFF file, including a multi-page / multi-channel stack
      * a .npy array
      * a PNG/JPEG file
      * a directory, in which case the first image found inside it is used
        (this covers Micro-Manager output such as `my_field/Pos0/img_000.tif`)

    For a stack, `frame` selects the page; without it the brightest page is
    used, which is the sensible default for a phase/fluorescence pair saved
    into one file.
    """
    path = Path(path)
    if path.is_dir():
        path = _first_image_in(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    else:
        from PIL import Image

        arr = np.array(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # A stack or an RGB image: reduce it to one plane.
        if arr.shape[-1] in (3, 4) and arr.ndim == 3 and arr.shape[0] > 4:
            # RGB(A) image - take the luminance.
            arr = arr[..., :3].mean(axis=-1)
        elif frame is not None:
            arr = arr[frame]
        else:
            # Pick the page with the most signal; for a phase/fluorescence pair
            # saved together this is the one the user means.
            arr = arr[int(np.argmax([np.percentile(p, 99.9) for p in arr]))]
    if arr.ndim != 2:
        raise ValueError(f"{path} does not contain a 2-D image (shape {arr.shape})")
    return arr.astype(np.float32)


def load_labels(path: str | Path) -> np.ndarray:
    """Load a label image (0 = background, 1..N = one integer per cell)."""
    arr = load_image(path)
    return arr.astype(np.int32)


def _first_image_in(directory: Path) -> Path:
    """Return the first image file inside a directory, searching sub-directories."""
    patterns = ("*.tif", "*.tiff", "*.npy", "*.png", "*.jpg", "*.jpeg")
    for pattern in patterns:
        hits = sorted(directory.glob(pattern)) or sorted(directory.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no image file found in {directory}")


def save_labels(path: str | Path, labels: np.ndarray) -> None:
    """Save a label image as .npy so it can be reused with --masks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels.astype(np.int32))
