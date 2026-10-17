# E. coli fluorescence localization and adhesion analysis

Image-analysis pipelines built for **iGEM Keio 2026 "One Direction"** and generalized
so that any lab with phase-contrast plus fluorescence images of rod-shaped bacteria
can run them on its own data.

Three questions, three commands:

| Pipeline | Question it answers | Command |
|---|---|---|
| **Polarity** | Is the fluorescence at one pole of each cell, and how does that change across an induction series? | `scripts/analyze_polarity.py` |
| **Adhesion** | Do cells of two differently labelled strains stick to each other? | `scripts/analyze_adhesion.py` |
| **Segmentation benchmark** | Do my images need a learned segmentation model, or is a threshold enough? | `scripts/benchmark_segmentation.py` |

Every pipeline takes plain command-line arguments, writes CSV tables and
publication-ready PNGs, and records the exact parameters it used in
`parameters.json` next to the results.

---

## Why these pipelines exist

Two problems turn up in almost every phase-contrast + fluorescence experiment on
bacteria, and both are handled here.

**Segmentation fails where the field is out of focus.** Classical thresholding
works on a well-focused field, but where cells are ringed by interference
fringes, Otsu marks background shading as cells, Sauvola leaves cell interiors
hollow and a texture filter inflates outlines out to the halo. These pipelines
use [Omnipose](https://github.com/kevinjohncutler/omnipose)'s `bact_phase_omni`
model, which was trained on rod-shaped bacteria and still resolves single cells
there. `benchmark_segmentation.py` reproduces that comparison on your images so
you can check the claim rather than take it on trust.

**Phase and fluorescence frames are not registered.** They are acquired one
after the other and do not land on the same pixels. In our data the offset
ranged from a few pixels to more than 120 px (about 20 um), differed between
the two fluorescence channels of one field, and varied by a few pixels across a
single field. Overlaying the images as acquired puts the signal outside the cell
outlines and makes any per-cell measurement meaningless. Both analysis pipelines
therefore search for the translation that maximises how much fluorescence falls
inside the cell masks — globally first, then locally.

---

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/igemkeio/ecoli-imaging
cd ecoli-imaging
pip install -r requirements.txt
```

Omnipose brings in PyTorch and is the heaviest dependency. If you only want to
try the pipelines on pre-computed masks, or you already segment your images with
something else, you can skip it:

```bash
pip install numpy scipy scikit-image pandas matplotlib tifffile pillow
```

and pass your label images with `--masks` (see below). A GPU is optional;
`--gpu` enables it, and segmenting a 2048 x 2048 field on a laptop CPU takes
roughly one minute.

### Check that it works, without any data of your own

```bash
python scripts/make_demo_data.py --out demo_data

python scripts/analyze_polarity.py \
    --field low  demo_data/low_phase.tif  demo_data/low_fluor.tif \
    --field mid  demo_data/mid_phase.tif  demo_data/mid_fluor.tif \
    --field high demo_data/high_phase.tif demo_data/high_fluor.tif \
    --masks low demo_data/low_masks.npy \
    --masks mid demo_data/mid_masks.npy \
    --masks high demo_data/high_masks.npy \
    --tile 256 --out demo_results
```

The demo images are synthetic: cells are drawn with a known polar focus and the
fluorescence is deliberately displaced from the phase image by a different
amount in each field. The run should recover those shifts to the pixel and
report a unipolar fraction that falls as the mock induction rises. `--masks`
supplies the ground-truth labels, so this works without Omnipose installed.

---

## Pipeline 1 - polar localization across an induction series

```bash
python scripts/analyze_polarity.py \
    --field "1 nM"   data/1nM_phase.tif   data/1nM_mcherry.tif \
    --field "10 nM"  data/10nM_phase.tif  data/10nM_mcherry.tif \
    --field "100 nM" data/100nM_phase.tif data/100nM_mcherry.tif \
    --pixel-size-um 0.1625 \
    --out results/popz
```

Repeat `--field LABEL PHASE FLUOR` for every field. Fields sharing a label are
pooled, so two fields of the same concentration simply get the same label.

**What it does, in order**

1. **Segment** the phase image with Omnipose. The model is run on the inverted
   and on the normal image and the two results are combined: slightly
   defocused cells appear bright-centred and are found in one, well-focused
   cells in the other. (`--polarity-variant` restricts this to one of them.)
2. **Clean the masks**: fill hollow masks, merge sliver fragments into the cell
   they belong to, drop cells touching the border.
3. **Correct the shift**: one global translation, then a refinement per tile
   (512 px by default), each maximising the fluorescence falling inside masks.
4. **Select cells to analyse** by size — by default length 2-4 um (maximum
   Feret diameter), width 0.8-1.2 um, area 1.4-4.2 um^2, which keeps normally
   grown, non-dividing cells. Cells outside the window stay in the mask image,
   so their fluorescence is still excluded from a neighbour's background, but
   they are flagged `strict_ok = False` and left out of the statistics.
5. **Call each cell**. With the default `--rule half`, a cell is **unipolar**
   when at least 90 % of its focus signal lies in one half of the long axis,
   **non-unipolar** otherwise, and **no fluorescence** when fewer than three
   pixels rise above the local background by 3 sigma.

**Why "one half" and not "the tip"**

The centre of an E. coli cell is occupied by the nucleoid, so a polar protein
accumulates towards the poles. When the cluster grows it pushes the nucleoid
aside and the signal extends from one pole towards mid-cell — still
localization to a single pole. A rule that only accepts signal in the outermost
25 % of the cell discards exactly those cells. Asking whether the signal stays
within one half keeps them, and the two rules agree on everything else.

**Outputs** in `--out`:

| File | Contents |
|---|---|
| `cells.csv` | one row per cell: position, size, background, signal, `half_frac`, `classification`, `strict_ok` |
| `fields.csv` | one row per field: cell counts, the shift that was applied, how much signal ended up inside masks |
| `summary.csv` | one row per condition, with Wilson 95 % confidence intervals |
| `figures/<field>_alignment.png` | before/after the shift correction — **look at this first** |
| `figures/<field>_classification.png` | every cell outlined in the colour of its call |
| `figures/composition.png`, `unipolar_fraction.png`, `expression_level.png` | the summary plots |
| `figures/alignment_shift_grid.png` | the per-tile shift of every field, as a quiver plot |
| `parameters.json` | the full command and every parameter used |

**Useful options**

```
--rule profile          four-way call (unipolar / bipolar / diffuse / none) instead of the default
--length-um 2 4         size window of the analysed cells; also --width-um, --area-um2
--no-size-filter        analyse every segmented cell
--half-min 0.90         fraction of focus signal that must lie in one half for "unipolar"
--signal-k 3.0          a pixel counts as signal at local background + k * sigma
--global-search 200     maximum shift searched, in pixels - raise it if your offset is larger
--tile 512              tile size of the local refinement; use 256 on small images
--no-align              skip the correction entirely (only if your channels are already registered)
--masks LABEL FILE      use a pre-computed label image instead of running the model
--save-masks            keep the computed masks so a re-run is instant
```

### A caveat about bipolar cells

`--rule profile` can report bipolar cells, but in a cell that is evenly filled,
small fluctuations at the two ends are easily read as two poles. In our own
data, automated bipolar counts at the highest induction came out at 69 % where
manual counting of the same crops gave 22 %. Treat bipolar counts from this
rule as a hypothesis to check by eye, not as a result. The default `half` rule
avoids the problem by not claiming to separate bipolar from delocalized cells:
both are "non-unipolar".

---

## Pipeline 2 - two-colour adhesion

```bash
python scripts/analyze_adhesion.py \
    --field mixed_100nM data/mix_phase.tif data/mix_red.tif data/mix_green.tif \
    --field single_red  data/red_phase.tif data/red_only.tif \
    --channel-names Nb2 Ag2 --channel-colors red green \
    --pixel-size-um 0.1625 \
    --out results/adhesion
```

A field given two images is phase + one channel; three images is phase + two
channels. Mixed cultures and their single-strain controls can go in the same run.

**What it does**

1. Segment the whole field as above.
2. Score every square position of a sliding window by the fraction of cells with
   fluorescence inside them, and keep the best `--top-k` windows that do not
   overlap at all (0 px^2 shared). The crops shown side by side are therefore
   genuinely different parts of the sample. If the field is too small or too
   sparse, fewer crops are returned rather than overlapping ones.
3. Align **each channel separately** inside each crop: a coarse binary search
   within `--max-shift` px, then a fine search on the raw image. The aligned
   crop is cut out of the full image at the shifted position, so it never gains
   a blank border.
4. Write a merged image, a before/after comparison, and a four-panel QC figure
   per crop, and count the cells positive for each channel and for both.

**Outputs**: `crops.csv` (one row per crop: position, shift per channel, how
much signal moved inside the cells, cell counts), `cells.csv`, `summary.csv`
with the double-positive fraction per field, and per-field folders of figures.

The number to compare between a mixed culture and its single-strain controls is
the double-positive fraction, together with the merged images: cells of two
different colours in contact is what adhesion looks like here.

---

## Pipeline 3 - segmentation benchmark

```bash
python scripts/benchmark_segmentation.py \
    --phase data/field1_phase.tif --phase data/field2_phase.tif \
    --crop-size 300 --out results/segmentation
```

Produces one figure per image: rows are regions, columns are raw phase contrast,
Global Otsu, Adaptive Sauvola, local texture and Omnipose. Without `--crop`,
the regions are chosen automatically where the methods disagree most, which is
where the choice of method actually matters. `--skip-omnipose` compares the
classical methods alone and needs no model. `segmentation_stats.csv` lists the
object count and median object area per method — a method that finds twice as
many objects as the others is usually segmenting the background.

---

## Using the modules directly

```python
from ecoli_imaging import imageio, segmentation, polarity
from ecoli_imaging.alignment import align_field

phase = imageio.load_image("phase.tif")
fluor = imageio.load_image("mcherry.tif")

labels, stats = segmentation.segment_phase(phase, enrichment_image=fluor)
result = align_field(fluor, labels)
cells = polarity.classify_cells(result.aligned,
                                segmentation.measurement_masks(labels),
                                rule="half")

print(sum(c["classification"] == "unipolar" for c in cells), "unipolar cells")
```

| Module | What it holds |
|---|---|
| `ecoli_imaging.imageio` | reading TIFF stacks, `.npy`, Micro-Manager folders |
| `ecoli_imaging.segmentation` | Omnipose wrapper, mask repair, size filtering |
| `ecoli_imaging.alignment` | whole-field and per-crop shift correction |
| `ecoli_imaging.polarity` | per-cell classification, summaries, Wilson intervals |
| `ecoli_imaging.crops` | confinement scoring and non-overlapping crop selection |
| `ecoli_imaging.classical` | Otsu / Sauvola / texture segmentation |
| `ecoli_imaging.viz` | every figure the scripts produce |

---

## Adapting it to your setup

**Pixel size.** `--pixel-size-um` is the one option you must set. Every size
limit is given in micrometres and converted with it, so an unset pixel size
silently filters out all your cells. Ours is 0.1625 um/px (40x objective,
6.5 um camera pixel).

**Cells of a different size or species.** Adjust `--length-um`, `--width-um`
and `--area-um2`. Run once with `--no-size-filter` and look at the `len_feret_um`
and `wid_um` columns of `cells.csv` to see where your population actually sits.

**Cells thinner than about 6 px.** Raise `--scale`: the model is applied to an
upscaled image because it was trained on larger cells.

**A very large or very small offset between channels.** `--global-search`
(pipeline 1) and `--max-shift` (pipeline 2) bound the search. Searching further
than necessary costs time and invites a wrong match on sparse fields.

**No offset at all.** `--no-align` leaves the images untouched. Check
`figures/<field>_alignment.png` before assuming this.

**Images that are already segmented.** `--masks` accepts any label image
(`.npy` or TIFF, 0 = background, one integer per cell), so Cellpose, Ilastik,
SuperSegger or hand-drawn masks all work and Omnipose is not needed.

---

## Checking that a result is real

Three things are worth doing before believing a number from this code.

1. **Look at the alignment figure.** If the signal still sits beside the cells
   after correction, nothing downstream is meaningful. A sparse field or a
   channel that is simply not expressed can send the search to a wrong peak;
   the reported in-mask fraction before and after is the quick check.
2. **Look at the classification figure.** Wrong calls are usually obvious.
3. **Count some cells by eye.** In our own analysis, manual counting confirmed
   where the unipolar maximum lay but corrected the automated split between
   bipolar and delocalized cells. `cells.csv` carries the per-cell position, so
   any cell in a figure can be traced back to its row.

Known limitations, stated plainly: a single field per condition gives no
biological replicate, however many cells it contains; the size window discards a
share of normal cells whose masks were inflated by defocus; dividing cells that
segment into two masks are not detected as such; and the `half` rule counts a
bright spot anywhere in one half of the cell as unipolar, including one sitting
at the edge of the nucleoid rather than at the pole.

---

## Citing

If this code is useful in your work, please cite the iGEM Keio 2026 team wiki
and this repository:

    iGEM Keio 2026. ecoli-imaging: polar localization and cell-cell adhesion
    analysis for E. coli microscopy. https://github.com/igemkeio/ecoli-imaging

The methods it implements are described, with the figures they produce, on the
team's Measurement page.

## License

MIT - see `LICENSE`.
