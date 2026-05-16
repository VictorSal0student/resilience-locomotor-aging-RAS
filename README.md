# Resilience Postural Salvat Victor

Pipeline for quantifying **locomotor resilience** in older adults (65+) during sand walking, under three auditory conditions: silence, random tempo, and adaptive beatMove.

Master 1 IEAP DigiMove-IDIL internship at EuroMov DHM Lab (Université de Montpellier), Oct 2025 – Aug 2026. Supervisors: A. Dufourneau, L. Damm, Pr. H. Blain.

## Method

- **Acquisition**: CODAMOTION optoelectronic system at 400 Hz, 12 markers (4 back + 4 left foot + 4 right foot)
- **Preprocessing**: sacrum barycentre (Dos01–04), MAD despike, adaptive EMD reconstruction (walking band 0.5–2.5 Hz), Butterworth low-pass 5 Hz, decimation to 100 Hz
- **Perturbation detection**: point-in-rectangle test against the measured sand bed coordinates
- **Resilience quantification**: rolling variance + phase-space radius + local cadence, compared between perturbation and corridor baseline passages
- **Statistics**: Friedman + pairwise Wilcoxon signed-rank (Bonferroni), per-metric filtering

## Project structure

```
.
├── src/resilience/         Reusable Python package
│   ├── config.py           Scientific constants
│   ├── paths.py            Disk paths (reads .env)
│   ├── participants.py     Participant registry (CRF + manual crops)
│   ├── io/                 Loading and saving
│   ├── processing/         Preprocess, sand detection, TDE primitives
│   ├── analysis/           High-level pipelines (TDE workflow, resilience workflow)
│   └── viz/                Plotting helpers + memoire figures
│
├── notebooks/              Orchestrator notebooks (one per pipeline stage)
│   ├── 00_tools.ipynb              Diagnostic toolbox (not run end-to-end)
│   ├── 01_batch_preprocess.ipynb   raw → processed
│   ├── 02_sand_detection.ipynb     Perturbation timestamps
│   ├── 03_tde.ipynb                Embedding parameters
│   ├── 04_resilience.ipynb         Resilience metrics
│   ├── 05_figures.ipynb            Publication figures
│   └── archive/                    Legacy notebooks (frozen)
│
├── scripts/
│   └── stats.R             Inter-condition statistics
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
| 5 | `scripts/stats.R` | `04_resilience_metrics.csv` | `outputs/tables/05_*.csv`, `outputs/figures/05_stats_boxplots.png` |
| 6 | `notebooks/05_figures.ipynb` | every previous output | `outputs/figures/fig{1..7}_*.png` |

## Setup

See [`SETUP.md`](./SETUP.md).

## Contact

Victor Salvat — victor.salvat@etu.umontpellier.fr
