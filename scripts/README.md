# R scripts

## Setup

```r
install.packages(c("tidyverse", "rstatix", "ggpubr", "here", "coin"))
```

R ≥ 4.2.0.

## Run

```bash
Rscript scripts/stats.R
```

## Files

| Script | Input | Outputs |
|---|---|---|
| `stats.R` | `outputs/tables/04_resilience_metrics.csv` | `outputs/tables/05_friedman.csv`, `05_wilcoxon_posthoc.csv`, `05_descriptive_stats.csv`, `outputs/figures/05_stats_boxplots.png` |
