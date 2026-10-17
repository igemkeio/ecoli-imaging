# Method notes

What each step actually computes, for readers who want to judge the numbers
rather than only run the code. Parameter names in parentheses are the
command-line options.

## 1. Segmentation

The `bact_phase_omni` Omnipose model is applied to the phase-contrast image,
upscaled by 1.5x first (`--scale`) because the cells are only about 6 px wide at
0.1625 um/px, below the scale the model was trained on. Masks are resized back
with nearest-neighbour interpolation so label values survive.

The model is run twice, on the normal and on the inverted image
(`--polarity-variant both`). Slightly defocused cells appear bright-centred and
are segmented from the inverted image; well-focused cells from the normal one.
The better run is chosen by *signal enrichment* — the fraction of bright
fluorescence pixels falling inside masks, divided by the area fraction the masks
cover, so that wide masks tracing phase halos are not rewarded for being large.
Cells from the other run are added only where they neither overlap nor come
within 2 px of an existing cell.

Clean-up, in order:

1. **Fill**: close each label with a disk (radius 3 px) and fill holes. On
   defocused images the model often traces only the dark cell outline, leaving
   a label shaped like two parallel stripes. Only background pixels are
   claimed, so neighbours are never absorbed.
2. **Merge slivers**: a fragment narrower than 0.55 um or smaller than
   0.4 um^2 is merged into the touching label it shares the most boundary with,
   but only if the union stays cell-shaped (width <= 1.6 um, solidity >= 0.8).
3. **Filter**: drop objects outside a generous size window (length 1-8 um,
   width 0.55-1.6 um, area 0.4-6 um^2) or touching the image border (5 px).

## 2. Shift correction

Let `M` be the binary cell mask and `B` the fluorescence image thresholded at
`median + 5 sigma` of the background, with background statistics taken outside
a dilated mask and sigma estimated from the median absolute deviation. The
chosen shift maximises

    sum over pixels of  M(y, x) * B(y - dy, x - dx)

computed for all offsets at once by FFT cross-correlation, searched within
`--global-search` px. The field is then divided into tiles (`--tile`) and each
tile is re-searched within `--tile-search` px of the global solution. Tiles with
fewer than 40 signal pixels keep the global shift, and a tile whose shift
differs from the median of its 3x3 neighbourhood by more than 6 px is replaced
by that median, so one noisy tile cannot tear the image apart.

The raw (not thresholded) fluorescence image is what gets translated, with
vacated border pixels filled by the background median rather than zero.

For crops in the adhesion pipeline the same idea runs in two stages
(`estimate_crop_shift`): a coarse search on the brightest 5 % of pixels within
`--max-shift`, then a fine search within `--fine-range` px that maximises the
raw fluorescence energy inside the masks. A channel whose maximum lies below
`background + 3 sigma` keeps a shift of (0, 0): a channel that is not expressed
must not be dragged onto the cells.

Reported quality measures: `inclusion_raw` and `inclusion_tile` are the fraction
of thresholded signal pixels inside masks before and after correction. If these
barely differ, the correction did nothing and the result should be inspected.

## 3. Per-cell measurement

For each cell, background is measured in a ring 3-10 px outside it that excludes
all other cells; cells with fewer than 30 ring pixels fall back to the global
background. Measurements are taken on masks expanded by 2 px, because a polar
focus sits on the membrane at the tip and optical blur (PSF FWHM ~0.4 um) plus
1-2 px of residual alignment error puts part of it just outside the
phase-contrast outline.

- **No fluorescence**: fewer than 3 pixels above `local background + 3 sigma`
  (`--signal-k`; the pixel count is `PolarityParams.min_signal_px` when calling
  the module directly).
- **Ambiguous** (excluded from statistics): the brightest pixel in a 3 px ring
  just outside the cell exceeds the brightest pixel inside. The signal then
  belongs to a neighbour's halo.
- **Focus pixels**: significant pixels at or above 40 % (`--focus-level`) of
  this cell's peak excess. Restricting the measurement to the bright core stops
  the blur around a polar spot from leaking into mid-cell and looking diffuse.
- **Unipolar** (`--rule half`): at least `--half-min` (0.90) of the focus signal
  weight lies in one half of the long axis. The long axis is the first principal
  component of the mask pixel coordinates.

The alternative `--rule profile` bins intensity into 20 bins along the long
axis, averages the outer ~15 % at each end, and calls a pole when it exceeds
mid-cell by a factor of 1.35; both poles above that threshold with the weaker at
least 0.55 of the stronger gives bipolar. See the caveat in the README: this
rule overcalls bipolar cells in evenly filled cells.

## 4. Crop selection

A window of `--crop-size` px slides over the field in `--stride` steps. Each
position is scored by the fraction of cells whose centroid lies inside it and
which carry signal above `background + 3.5 sigma` in at least one channel.
Positions are ranked by that fraction, then by the number of cells carrying
every channel, then by labelled cell count. Crops are accepted greedily and a
candidate sharing even one pixel with an accepted crop is skipped, so the
selected regions are strictly disjoint.

## 5. Statistics

Proportions are reported with Wilson 95 % confidence intervals rather than the
normal approximation, because per-condition counts are small and the
proportions sit close to 0 or 1. The denominator matters: `unipolar_pct` is over
all classified cells, `unipolar_among_fluorescent_pct` over fluorescent cells
only. The second is the one to plot against induction; the first falls simply
because fewer cells express.

All of this describes within-sample variation. A single field gives no
biological replicate however many cells it contains.
