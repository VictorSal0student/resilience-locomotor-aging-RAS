# =============================================================================
# stats_dynamics.R — Inter-condition statistics for resilience DYNAMICS metrics
# =============================================================================
#
# Companion to scripts/stats.R (which handles resilience metrics from 04).
# This script analyses the temporal-dynamics metrics from 04b:
#   - lag_time_s
#   - peak_time_s
#   - recovery_time_s
#   - peak_distance_mm
#   - n_steps_recovery
#   - persistent_diff_15s / 30s / 45s / 60s
#   - pct_T1 / pct_T2 / pct_T3 (pre-perturbation stability profile)
#
# Friedman test (omnibus, paired) + pairwise Wilcoxon signed-rank
# with three correction levels (uncorrected, Holm, Bonferroni) and
# effect sizes (Wilcoxon r).
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(rstatix)
  library(readr)
  library(stringr)
  library(here)
  library(purrr)
})

# ── Paths ────────────────────────────────────────────────────────────────────
project_root <- here::here()
input_csv    <- file.path(project_root, "outputs", "tables",
                            "04b_resilience_dynamics.csv")
output_dir   <- file.path(project_root, "outputs", "tables")
figures_dir  <- file.path(project_root, "outputs", "figures")
dir.create(output_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading:", input_csv, "\n")
df <- read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(condition = factor(condition,
                              levels = c("silence", "tempo_random",
                                          "beatmove_adaptatif")))

cat("N rows:", nrow(df), "  |  N participants:", n_distinct(df$participant),
    "  |  N conditions per participant:\n")
print(table(df$participant, df$condition))

# ── Metrics to analyse ───────────────────────────────────────────────────────
metric_cols <- c(
  "lag_time_s", "peak_time_s", "recovery_time_s",
  "peak_distance_mm", "n_steps_recovery",
  "persistent_diff_15s", "persistent_diff_30s",
  "persistent_diff_45s", "persistent_diff_60s",
  "pct_T1", "pct_T2", "pct_T3"
)

# Keep only metrics present in the CSV
metric_cols <- metric_cols[metric_cols %in% names(df)]

# =============================================================================
# 1. DESCRIPTIVE STATS
# =============================================================================

cat("\n========================================================\n")
cat("DESCRIPTIVE STATS BY CONDITION\n")
cat("========================================================\n")

desc_stats <- df %>%
  pivot_longer(all_of(metric_cols), names_to = "metric", values_to = "value") %>%
  group_by(metric, condition) %>%
  summarise(
    N      = sum(!is.na(value)),
    mean   = round(mean(value, na.rm = TRUE), 3),
    sd     = round(sd(value,   na.rm = TRUE), 3),
    median = round(median(value, na.rm = TRUE), 3),
    q25    = round(quantile(value, 0.25, na.rm = TRUE), 3),
    q75    = round(quantile(value, 0.75, na.rm = TRUE), 3),
    min    = round(min(value, na.rm = TRUE), 3),
    max    = round(max(value, na.rm = TRUE), 3),
    .groups = "drop"
  )

print(desc_stats, n = Inf)
write_csv(desc_stats, file.path(output_dir, "05b_descriptive_stats.csv"))

# =============================================================================
# 2. FRIEDMAN (omnibus, paired)
# =============================================================================

cat("\n========================================================\n")
cat("FRIEDMAN TEST (paired, N participants with 3 conditions)\n")
cat("========================================================\n")

friedman_results <- map_dfr(metric_cols, function(metric) {
  d_wide <- df %>%
    select(participant, condition, all_of(metric)) %>%
    pivot_wider(names_from = condition, values_from = all_of(metric))

  # Keep only participants with all 3 conditions non-NA
  complete <- d_wide %>%
    filter(!is.na(silence) & !is.na(tempo_random) & !is.na(beatmove_adaptatif))

  if (nrow(complete) < 3) {
    return(data.frame(metric = metric, n = nrow(complete),
                       chi2 = NA, df = NA, p = NA, kendall_W = NA, sig = ""))
  }

  d_long <- complete %>%
    pivot_longer(cols = c(silence, tempo_random, beatmove_adaptatif),
                  names_to = "condition", values_to = "value") %>%
    mutate(condition = factor(condition,
                                levels = c("silence", "tempo_random",
                                            "beatmove_adaptatif")))

  ft <- friedman_test(d_long, value ~ condition | participant)
  W  <- friedman_effsize(d_long, value ~ condition | participant)

  sig <- case_when(
    ft$p < 0.001 ~ "***",
    ft$p < 0.01  ~ "**",
    ft$p < 0.05  ~ "*",
    ft$p < 0.10  ~ ".",
    TRUE          ~ "ns"
  )

  data.frame(
    metric    = metric,
    n         = nrow(complete),
    chi2      = round(as.numeric(ft$statistic), 2),
    df        = ft$df,
    p         = round(ft$p, 4),
    kendall_W = round(as.numeric(W$effsize), 3),
    sig       = sig
  )
})

print(friedman_results)
write_csv(friedman_results, file.path(output_dir, "05b_friedman.csv"))

# =============================================================================
# 3. WILCOXON POST-HOC (uncorrected, Holm, Bonferroni + effect size r)
# =============================================================================

cat("\n========================================================\n")
cat("WILCOXON SIGNED-RANK POST-HOC (uncorrected + Holm + Bonferroni)\n")
cat("========================================================\n")

wilcoxon_results <- map_dfr(metric_cols, function(metric) {
  d_wide <- df %>%
    select(participant, condition, all_of(metric)) %>%
    pivot_wider(names_from = condition, values_from = all_of(metric))

  complete <- d_wide %>%
    filter(!is.na(silence) & !is.na(tempo_random) & !is.na(beatmove_adaptatif))

  if (nrow(complete) < 3) return(NULL)

  d_long <- complete %>%
    pivot_longer(cols = c(silence, tempo_random, beatmove_adaptatif),
                  names_to = "condition", values_to = "value") %>%
    mutate(condition = factor(condition,
                                levels = c("silence", "tempo_random",
                                            "beatmove_adaptatif")))

  pw_bonf <- d_long %>%
    wilcox_test(value ~ condition, paired = TRUE,
                p.adjust.method = "bonferroni") %>%
    add_significance("p.adj")

  pw_holm <- d_long %>%
    wilcox_test(value ~ condition, paired = TRUE,
                p.adjust.method = "holm") %>%
    add_significance("p.adj")

  es <- d_long %>%
    wilcox_effsize(value ~ condition, paired = TRUE)

  pw_bonf %>%
    left_join(pw_holm %>% select(group1, group2,
                                   p_holm = p.adj, sig_holm = p.adj.signif),
              by = c("group1", "group2")) %>%
    left_join(es %>% select(group1, group2, effsize, magnitude),
              by = c("group1", "group2")) %>%
    transmute(metric        = metric,
              pair          = paste(substr(group1, 1, 3), "vs",
                                    substr(group2, 1, 3)),
              N             = n1,
              statistic     = round(statistic, 1),
              p_uncorrected = round(p, 4),
              p_holm        = round(p_holm, 4),
              sig_holm      = sig_holm,
              p_bonf        = round(p.adj, 4),
              sig_bonf      = p.adj.signif,
              r             = round(effsize, 3),
              magnitude     = magnitude)
})

print(wilcoxon_results, n = Inf)
write_csv(wilcoxon_results, file.path(output_dir, "05b_wilcoxon_posthoc.csv"))

# =============================================================================
# 4. BOXPLOTS PAR METRIQUE
# =============================================================================

cat("\n========================================================\n")
cat("BOXPLOTS\n")
cat("========================================================\n")

cond_colors <- c(silence            = "#4F81BD",
                  tempo_random       = "#E69F00",
                  beatmove_adaptatif = "#3CB371")

# Subset of the most informative metrics for the main figure
plot_metrics <- c("lag_time_s", "peak_time_s", "recovery_time_s",
                   "peak_distance_mm", "persistent_diff_30s",
                   "pct_T1", "pct_T2", "pct_T3")
plot_metrics <- plot_metrics[plot_metrics %in% metric_cols]

plots <- map(plot_metrics, function(metric) {
  fr <- friedman_results %>% filter(metric == !!metric)
  subtitle <- if (nrow(fr) == 1 && !is.na(fr$p)) {
    sprintf("N = %d  |  Friedman chi2 = %.2f, p = %.3f %s  |  W = %.2f",
             fr$n, fr$chi2, fr$p, fr$sig, fr$kendall_W)
  } else { "" }

  sig_pairs <- wilcoxon_results %>%
    filter(metric == !!metric, sig_holm != "ns")

  p <- ggplot(df, aes(x = condition, y = .data[[metric]], fill = condition)) +
    geom_boxplot(alpha = 0.5, outlier.shape = NA) +
    geom_jitter(aes(color = condition), width = 0.12, size = 1.8, alpha = 0.8) +
    geom_line(aes(group = participant), color = "grey60",
              alpha = 0.5, linewidth = 0.3) +
    scale_fill_manual (values = cond_colors) +
    scale_color_manual(values = cond_colors) +
    labs(title    = str_replace_all(metric, "_", " "),
         subtitle = subtitle,
         x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none",
          plot.subtitle   = element_text(size = 9, color = "grey30"))
  p
})

# Combine plots into a grid figure
n_plots <- length(plots)
ncol <- min(4, n_plots)
fig <- patchwork::wrap_plots(plots, ncol = ncol)

ggsave(file.path(figures_dir, "05b_stats_boxplots.png"),
        fig, width = 5 * ncol, height = 4 * ceiling(n_plots / ncol), dpi = 120)

cat("\nFigure saved:", file.path(figures_dir, "05b_stats_boxplots.png"), "\n")

# =============================================================================
# 5. RESUME
# =============================================================================

cat("\n========================================================\n")
cat("EXPORT FILES\n")
cat("========================================================\n")
cat(" 05b_descriptive_stats.csv\n")
cat(" 05b_friedman.csv\n")
cat(" 05b_wilcoxon_posthoc.csv\n")
cat(" 05b_stats_boxplots.png\n")

cat("\nFriedman significant metrics (p < 0.05):\n")
print(friedman_results %>% filter(p < 0.05))

cat("\nWilcoxon pairs surviving Holm correction:\n")
print(wilcoxon_results %>% filter(sig_holm != "ns"))

cat("\nWilcoxon pairs surviving Bonferroni correction:\n")
print(wilcoxon_results %>% filter(sig_bonf != "ns"))

cat("\nDONE\n")
