# Beyond the Color Wheel: Validating a Depth-Aware Computational Measure of Spatial Color Harmony in Interior Scenes

Code, manual annotations, per-image data and outputs for the article

> Bal, H., Koyuncu, F., & Polat, A. I. (2026). *Beyond the color wheel: Validating a depth-aware computational measure of spatial color harmony in interior scenes.* Manuscript submitted for publication.

The repository contains everything needed to reproduce the reported numbers, tables and figures: the analysis script, the
manual interior annotations, the identifiers and per-image features of every analyzed image, and the outputs of the
published run.

## What the study does

A depth-aware color harmony score is computed for interior photographs: a monocular depth network (MiDaS) partitions each
image into three depth layers, a dominant color is extracted from each layer, and the complete two-color harmony model of
Ou and Luo (2006) is applied between the layers. The score is tested in three studies:

| Study | Question | Data |
|---|---|---|
| 1 | Does the score track human color harmony ratings, and does the depth partition add information beyond depth-free partitions and basic color statistics? | AADB (Kong et al., 2016): 234 manually verified interior photographs, 578 CLIP candidates, 1,500 general images |
| 2 | Can monocular depth replace measured depth for this purpose? | 300 SUN RGB-D scenes with sensor depth (Song et al., 2015) |
| 3 | How are colorfulness, depth structure and the harmony score related within interior corpora? | The same SUN RGB-D scenes and 300 images of the Interior-Style corpus |

## Repository contents

```
beyond_color_wheel.py          complete analysis pipeline (both modes, see below)
make_annotation_sheets.py      creates the sheets used for the manual interior annotation
requirements.txt               packages for the reanalysis mode
requirements-full.txt          additional packages for the full pipeline
annotations/                   manual interior labels (first annotator, 578 images) and second_rater/ (100 images)
outputs/
  figures/                     main figures (PNG 300 dpi and PDF)
  tables/                      main tables (CSV and XLSX)
  supplementary/               supplementary tables and figures
  data/                        per-image features with image identifiers
  reproducibility/             configuration, software environment, package versions, key results
CITATION.cff, LICENSE
```

## Reproducing the results

### Option A: reanalysis from the shared per-image features (recommended for checking the results)

Needs Python 3.10 or newer; no GPU and no image data. Runs in a few minutes.

```
python -m venv venv
venv\Scripts\activate              (Windows)   or   source venv/bin/activate   (Linux/macOS)
python -m pip install -r requirements.txt
python beyond_color_wheel.py --from-features
```

The script reads `outputs/data/*_per_image_features.csv` and the files in `annotations/`, and recomputes every statistic,
table and figure that does not display images. Running it on the files of this repository reproduces the published values
exactly. Figure 1, the image contact sheets and the AADB label-matching table require the image data and are produced by
the full pipeline only.

### Option B: full pipeline (downloads the data, runs CLIP and MiDaS)

Tested on Windows 10 with Python 3.14.8 and an NVIDIA GeForce RTX 3080 (PyTorch 2.14.1, CUDA 13.0); it also runs on
Linux/macOS and on the CPU (slower).

```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-full.txt
python beyond_color_wheel.py
```

On Windows, `pip install torch` from PyPI installs a CPU-only build; the command above installs the CUDA build. On Linux or
macOS, `pip install torch torchvision` is sufficient.

The script downloads AADB from the Hugging Face Hub (`Iceclear/AADB`, about 2.6 GB), SUN RGB-D V1 from the project server
(about 6.5 GB) and streams the Interior-Style corpus (`hammer888/interior_style_dataset`). Downloads and intermediate
results are cached in `data/`, `cache_hf/`, `cache_torch/` and `outputs/cache/`, so an interrupted run can simply be
restarted. All settings are defined in the `CONFIG` block at the top of the script; a fixed seed (42) and per-image random
generators make the results independent of processing order and caching.

Manual annotation: the verified interior subset is defined by the labels in `annotations/interior_annotation.xlsx`. To
repeat the annotation from scratch, run the full pipeline once, then `python make_annotation_sheets.py`, label the numbered
pages following the criteria below, and run the full pipeline again.

## Manual interior annotation

All 578 CLIP candidates were labeled by the second author; a random subsample of 100 was labeled independently by another
author (agreement 94/100, Cohen's kappa = 0.86). The annotation sheets showed numbered thumbnails only, so ratings, CLIP
probabilities and harmony scores were not visible. The criteria (Table S2 of the article) are listed in
[`annotations/README.md`](annotations/README.md) and in the `criteria` sheet of each workbook.

## Where to find each item of the article

| Article | File |
|---|---|
| Table 1 | `outputs/tables/Table1_study_samples` |
| Table 2 | `outputs/tables/Table3_aadb_correlations` (outcome = ColorHarmony) |
| Table 3 | `outputs/tables/Table4_aadb_depth_vs_controls`, `Table5_aadb_partial_correlations`, `Table6_aadb_incremental_validity` |
| Table 4 | `outputs/tables/Table8_sun_depth_agreement` |
| Table 5 | `outputs/tables/Table10_within_corpus_correlations` |
| Figure 1 | `outputs/figures/Fig1_pipeline_examples` |
| Figure 2 | `outputs/figures/Fig3_correlations_with_human_ratings` |
| Figure 3 | `outputs/figures/Fig4_depth_vs_control_differences` |
| Figure 4 | `outputs/figures/Fig7_sun_depth_agreement` |
| Figure 5 | `outputs/figures/Fig8_sun_harmony_agreement` |
| Figure 6 | `outputs/figures/Fig9_within_corpus_colorfulness_depth` |
| Table S1 | `outputs/supplementary/TableS2_aadb_label_matching` |
| Table S2 | `annotations/README.md` (annotation criteria) |
| Table S3 | `outputs/supplementary/TableS1_ou_luo_implementation_checks` |
| Table S4 | `outputs/supplementary/TableS3_aadb_quality_control` |
| Table S5 | `outputs/supplementary/TableS6_interior_annotation_summary` |
| Table S6 | `outputs/tables/Table2_aadb_descriptives` |
| Table S7 | `outputs/tables/Table7_aadb_sensitivity` |
| Table S8 | `outputs/tables/Table3_aadb_correlations` (outcome = AestheticScore) |
| Table S9 | `outputs/supplementary/TableS4_sun_agreement_by_sensor` |
| Table S10 | `outputs/tables/Table9_corpus_descriptives` |
| Table S11 | `outputs/tables/Table12_corpus_differences` |
| Table S12 | `outputs/tables/Table11_interaction_model` |
| Table S13 | `outputs/tables/Table13_within_corpus_categories` |
| Figure S1 | `outputs/supplementary/FigS4_aadb_rating_distribution` |
| Figures S2–S5 | image contact sheets, produced by the full pipeline (`FigS1`, `FigS5`, `FigS2`, `FigS3`); not distributed here, see below |
| Figure S6 | `outputs/figures/Fig2_Hdepth_by_human_rating` |
| Figure S7 | `outputs/figures/Fig6_incremental_validity` |
| Figure S8 | `outputs/figures/Fig5_sensitivity_K_and_weighting` |
| Figure S9 | `outputs/figures/Fig10_corpus_distributions` |
| Figure S10 | `outputs/figures/Fig11_within_corpus_categories` |

## Per-image data

`outputs/data/` lists every analyzed image with its identifier and all computed features:

- `aadb_per_image_features`: AADB image key and archive path, human Color Harmony and aesthetic scores, CLIP probabilities,
  sample membership, manual interior label, color management flags, color statistics and all harmony variants.
- `aadb_scene_filter_and_samples`: CLIP probabilities and sample membership for all 9,958 AADB images with a Color Harmony label.
- `sun_per_image_features`: SUN RGB-D scene path, sensor, room type, depth-agreement measures and features.
- `style_per_image_features`: Interior-Style image hash (first 16 hex digits of the MD5 of the decoded image), position in
  the shuffled stream, style label and features.

Main variables: `H_depth` (depth-aware harmony, K = 3, equal weights), `H_depth_K2` to `H_depth_K5`, `H_depth_adjw`,
`H_depth_areaw`, `H_global`, `H_bands`, `H_shuffled`, `colorfulness` (Hasler & Süsstrunk, 2003), `color_entropy_bits`,
`mean_L`, `mean_C`, `edge_density` and `depth_dispersion`.

## Data sources and licenses

No images are redistributed in this repository. The image data must be obtained from the original sources, under their own
terms of use:

- AADB: Kong, S., Shen, X., Lin, Z., Mech, R., & Fowlkes, C. (2016). Photo aesthetics ranking network with attributes and
  content adaptation. In *Computer Vision – ECCV 2016* (pp. 662–679). Springer. https://doi.org/10.1007/978-3-319-46448-0_40
- SUN RGB-D: Song, S., Lichtenberg, S. P., & Xiao, J. (2015). SUN RGB-D: A RGB-D scene understanding benchmark suite. In
  *Proceedings of the IEEE CVPR* (pp. 567–576). https://doi.org/10.1109/CVPR.2015.7298655
- Interior-Style corpus: Hugging Face dataset `hammer888/interior_style_dataset` (see its dataset card).

The image contact sheets (Figures S2–S5) and the thumbnails written by the full pipeline show images from these datasets
and are therefore not included; the full pipeline recreates them.

The code is released under the MIT License (see `LICENSE`). The manual annotations and the derived per-image data are
released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Citation

Please cite the article and this repository (see `CITATION.cff`).
