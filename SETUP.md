# Setup

## Requirements

- Python ≥ 3.11
- R ≥ 4.2 (for `scripts/stats.R` and `scripts/stats_dynamics.R`)
- Conda recommended for the Python environment

## Python environment

```bash
conda create -n resilience python=3.11
conda activate resilience

pip install -e .
```

Installs the `resilience` package and its dependencies (numpy, scipy, pandas, matplotlib, scikit-learn, EMD-signal, python-dotenv, h5py).

## Environment variables (`.env`)

Disk paths are not hardcoded — they are read from a `.env` file at the project root. Copy the template and edit it:

```bash
cp .env.example .env
```

Then edit `.env`:

```
PROJECT_ROOT=/absolute/path/to/resilience-locomotor-aging-RAS
RAW_DIR=/path/to/data/raw
PROCESSED_DIR=/path/to/data/processed
BEATMOVE_DIR=/path/to/data/beatmove
EXTERNAL_DIR=/path/to/data/external
OUTPUTS_DIR=/path/to/outputs
```

The pipeline expects this naming convention inside `RAW_DIR`:

```
RAW_DIR/{participant_code}/{participant_code}_{NN}_{condition}.mat
```

Example: `RAW_DIR/001CrMa/001CrMa_01_silence.mat`.

## R environment (for `stats.R` and `stats_dynamics.R`)

From an R session:

```r
install.packages(c("tidyverse", "rstatix", "ggpubr", "here", "coin", "purrr"))
```

Note: `purrr` must be loaded explicitly (`library(purrr)`) in the scripts — it is not auto-loaded by `tidyverse` in all environments.

## Verify the install

```python
from resilience import paths
paths.check_dirs(verbose=True)
```

Should print `✅` next to every configured directory.

## Run the full pipeline

```bash
# Activate Python env, then run notebooks in order
jupyter lab notebooks/

# Run R stats (after notebook 04)
Rscript scripts/stats.R

# Run R stats dynamics (after notebook 04b)
Rscript scripts/stats_dynamics.R
```
