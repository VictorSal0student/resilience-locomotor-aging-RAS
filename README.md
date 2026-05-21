# Resilience Locomotor Aging — Rhythmic Auditory Stimulation

Pipeline for quantifying **locomotor resilience** in older adults (65+) during sand walking, under three auditory conditions: silence, random tempo, and adaptive beatMove.

Master 1 IEAP DigiMove-IDIL internship at EuroMov DHM Lab (Université de Montpellier), Oct 2025 – Aug 2026. Supervisors: A. Dufourneau, L. Damm, Pr. H. Blain.

## Method

- **Acquisition**: CODAMOTION optoelectronic system at 400 Hz, 12 markers (4 back + 4 left foot + 4 right foot)
- **Preprocessing**: sacrum barycentre (Dos01–04), MAD despike, adaptive EMD reconstruction (walking band 0.5–2.5 Hz), Butterworth low-pass 5 Hz, decimation to 100 Hz
- **Perturbation detection**: point-in-rectangle test (BBOX) against the measured sand bed coordinates (4.4 m × 1.0 m)
- **Resilience — characterisation (notebook 04)**: rolling variance + phase-space radius + local cadence, compared between perturbation and lateral baseline passages
- **Resilience — dynamics (notebook 04b)**: torus method (Wurdeman 2016, Ravi 2021) ported from MATLAB — lag, peak, recovery, persistence metrics
- **Statistics**: Friedman + pairwise Wilcoxon signed-rank (Holm + Bonferroni), per-metric filtering

## ⚠️ Data note — `raw_Dos` arrays are POST-crop

The arrays `raw_Dos01`–`raw_Dos04` stored in each `.npz` file are **already post-crop** (the first 60 s have been removed upstream by the preprocessor), despite their "raw" name. Timestamps derived from these arrays (`i / FS_RAW`) are therefore in post-crop reference and match `time_final` directly. **Do not subtract `crop_start_s` from these timestamps.**

## Project structure

```
.
├── src/resilience/         Reusable Python package
│   ├── config.py           Scientific constants (BBOX corners, margins, FS)
│   ├── paths.py            Disk paths (reads .env)
│   ├── participants.py     Participant registry (CRF + manual crops)
│   ├── io/                 Loading and saving
│   ├── processing/         Preprocess, sand detection, TDE primitives
│   ├── analysis/           High-level pipelines (TDE, resilience characterisation, resilience dynamics)
│   └── viz/                Plotting helpers + memoire figures
│
├── notebooks/              Orchestrator notebooks (one per pipeline stage)
│   ├── 00_tools.ipynb                  Diagnostic toolbox (not run end-to-end)
│   ├── 01_batch_preprocess.ipynb       raw → processed
│   ├── 02_sand_detection.ipynb         Perturbation timestamps
│   ├── 03_tde.ipynb                    Embedding parameters
│   ├── 04_resilience.ipynb             Resilience characterisation metrics
│   ├── 04b_resilience_dynamics.ipynb   Resilience dynamics (torus method)
│   └── 05_figures.ipynb                Publication figures
│
├── scripts/
│   ├── stats.R             Inter-condition statistics (notebook 04)
│   └── stats_dynamics.R    Inter-condition statistics (notebook 04b)
│
├── data/                   (gitignored)
│   ├── raw/                .mat files from CODAMOTION
│   ├── processed/          .npz files produced by notebook 01
│   ├── beatmove/           beatMove session logs
│   └── external/           CRF annotations, recap files
│
└── outputs/
    ├── tables/             CSV results
    └── figures/            PNG figures
```

## Pipeline (run in order)

| Step | File | Reads | Writes |
|---|---|---|---|
| 1 | `notebooks/01_batch_preprocess.ipynb` | `data/raw/` | `data/processed/*.npz` |
| 2 | `notebooks/02_sand_detection.ipynb` | `data/processed/` | `outputs/tables/02_sand_detection.csv` |
| 3 | `notebooks/03_tde.ipynb` | `data/processed/` | `outputs/tables/03_tde_*.csv`, `outputs/figures/03_*.png` |
| 4 | `notebooks/04_resilience.ipynb` | `data/processed/`, `03_tde_per_participant.csv` | `outputs/tables/04_resilience_metrics.csv`, `outputs/figures/04_*.png` |
| 4b | `notebooks/04b_resilience_dynamics.ipynb` | `data/processed/`, `03_tde_per_participant.csv` | `outputs/tables/04b_resilience_dynamics.csv`, `outputs/figures/04b_*.png` |
| 5 | `scripts/stats.R` | `04_resilience_metrics.csv` | `outputs/tables/05_*.csv`, `outputs/figures/05_stats_boxplots.png` |
| 5b | `scripts/stats_dynamics.R` | `04b_resilience_dynamics.csv` | `outputs/tables/05b_*.csv`, `outputs/figures/05b_stats_boxplots.png` |
| 6 | `notebooks/05_figures.ipynb` | every previous output | `outputs/figures/fig{1..7}_*.png` |

## Setup

See [`SETUP.md`](./SETUP.md).

## Contact

Victor Salvat — victor.salvat@etu.umontpellier.fr
