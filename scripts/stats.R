# =============================================================================
# stats.R
# =============================================================================
# Inter-condition statistics on resilience metrics.
#
# Design   : within-subject repeated measures (3 conditions per participant)
# Omnibus  : Friedman test + Kendall's W
# Post-hoc : pairwise Wilcoxon signed-rank
#            - Bonferroni (conservative, primary)
#            - Holm-Bonferroni (less conservative, complementary)
#            - uncorrected p-values + effect sizes also reported
# Effect   : r = Z / sqrt(N) via rstatix::wilcox_effsize
# Filter   : per metric, keep only participants with all 3 conditions valid
#
# Run      : Rscript scripts/stats.R
# Input    : outputs/tables/04_resilience_metrics.csv
# Outputs  : outputs/tables/05_friedman.csv
#            outputs/tables/05_wilcoxon_posthoc.csv
#            outputs/tables/05_descriptive_stats.csv
#            outputs/figures/05_stats_boxplots.png
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(rstatix)
  library(ggpubr)
  library(here)
})

# ---- Paths -------------------------------------------------------------------
PROJECT_ROOT <- here::here()
INPUT_CSV    <- file.path(PROJECT_ROOT, "outputs", "tables", "04_resilience_metrics.csv")
OUT_TABLES   <- file.path(PROJECT_ROOT, "outputs", "tables")
OUT_FIGURES  <- file.path(PROJECT_ROOT, "outputs", "figures")
dir.create(OUT_TABLES,  showWarnings = FALSE, recursive = TRUE)
dir.create(OUT_FIGURES, showWarnings = FALSE, recursive = TRUE)

# ---- Config ------------------------------------------------------------------
METRICS    <- c("z_var", "z_radius", "ratio_var", "ratio_radius", "delta_cadence_hz")
COND_ORDER <- c("silence", "tempo_random", "beatmove_adaptatif")
COND_COLORS <- c(silence            = "steelblue",
                 tempo_random       = "darkorange",
                 beatmove_adaptatif = "forestgreen")

# ---- Load --------------------------------------------------------------------
df <- read_csv(INPUT_CSV, show_col_types = FALSE) %>%
  mutate(condition = factor(condition, levels = COND_ORDER))

cat(sprintf("Loaded %d trials, %d participants\n",
            nrow(df), n_distinct(df$participant)))

# ---- 1. Friedman -------------------------------------------------------------
friedman_results <- purrr::map_dfr(METRICS, function(metric) {

  d_long <- df %>%
    select(participant, condition, all_of(metric)) %>%
    rename(value = all_of(metric)) %>%
    drop_na(value)

  complete_p <- d_long %>%
    count(participant) %>%
    filter(n == 3) %>%
    pull(participant)

  d_long <- d_long %>% filter(participant %in% complete_p)
  n <- length(complete_p)

  if (n < 3) {
    return(tibble(metric = metric, N = n, chi2 = NA_real_, df = NA_integer_,
                  p = NA_real_, kendall_W = NA_real_, sig = "n_too_small"))
  }

  fr  <- d_long %>% friedman_test(value ~ condition | participant)
  kw  <- d_long %>% friedman_effsize(value ~ condition | participant)

  sig <- case_when(fr$p < 0.001 ~ "***",
                   fr$p < 0.01  ~ "**",
                   fr$p < 0.05  ~ "*",
                   TRUE         ~ "ns")

  tibble(metric    = metric,
         N         = n,
         chi2      = round(fr$statistic, 3),
         df        = fr$df,
         p         = round(fr$p, 4),
         kendall_W = round(kw$effsize, 3),
         sig       = sig)
})

cat("\n=== FRIEDMAN ===\n")
print(friedman_results)
write_csv(friedman_results, file.path(OUT_TABLES, "05_friedman.csv"))

# ---- 2. Wilcoxon post-hoc ----------------------------------------------------
wilcoxon_results <- purrr::map_dfr(METRICS, function(metric) {

  d_long <- df %>%
    select(participant, condition, all_of(metric)) %>%
    rename(value = all_of(metric)) %>%
    drop_na(value)

  complete_p <- d_long %>%
    count(participant) %>%
    filter(n == 3) %>%
    pull(participant)

  d_long <- d_long %>% filter(participant %in% complete_p)

  if (length(complete_p) < 3) return(NULL)

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

cat("\n=== WILCOXON POST-HOC ===\n")
print(wilcoxon_results, n = 30)
write_csv(wilcoxon_results, file.path(OUT_TABLES, "05_wilcoxon_posthoc.csv"))

# ---- 3. Descriptive ----------------------------------------------------------
descriptive_stats <- purrr::map_dfr(METRICS, function(metric) {

  d_long <- df %>%
    select(participant, condition, all_of(metric)) %>%
    rename(value = all_of(metric)) %>%
    drop_na(value)

  complete_p <- d_long %>%
    count(participant) %>%
    filter(n == 3) %>%
    pull(participant)

  d_long <- d_long %>% filter(participant %in% complete_p)

  d_long %>%
    group_by(condition) %>%
    summarise(N      = n(),
              median = round(median(value), 3),
              iqr    = round(IQR(value),    3),
              mean   = round(mean(value),   3),
              sd     = round(sd(value),     3),
              .groups = "drop") %>%
    mutate(metric = metric, .before = condition)
})

cat("\n=== DESCRIPTIVE ===\n")
print(descriptive_stats, n = 30)
write_csv(descriptive_stats, file.path(OUT_TABLES, "05_descriptive_stats.csv"))

# ---- 4. Boxplots -------------------------------------------------------------
metrics_labels <- c(
  z_var            = "Z-score variance",
  z_radius         = "Z-score radius",
  ratio_var        = "Ratio var (pert / baseline)",
  ratio_radius     = "Ratio radius (pert / baseline)",
  delta_cadence_hz = "Delta cadence (Hz)"
)

make_plot <- function(metric) {

  d_long <- df %>%
    select(participant, condition, all_of(metric)) %>%
    rename(value = all_of(metric)) %>%
    drop_na(value)

  complete_p <- d_long %>%
    count(participant) %>%
    filter(n == 3) %>%
    pull(participant)

  d_long <- d_long %>% filter(participant %in% complete_p)
  n <- length(complete_p)

  sig_pairs <- wilcoxon_results %>%
    filter(metric == !!metric, sig_holm != "ns")

  fr_row  <- friedman_results %>% filter(metric == !!metric)
  subtitle <- sprintf("N = %d   |   Friedman chi2 = %.2f, p = %.3f %s   |   W = %.2f",
                      n, fr_row$chi2, fr_row$p, fr_row$sig, fr_row$kendall_W)

  p <- ggplot(d_long, aes(x = condition, y = value, fill = condition)) +
    geom_boxplot(width = 0.55, alpha = 0.4, outlier.shape = NA) +
    geom_line(aes(group = participant), color = "gray60",
              alpha = 0.4, linewidth = 0.4) +
    geom_point(aes(color = condition), size = 2, alpha = 0.8) +
    scale_fill_manual(values = COND_COLORS) +
    scale_color_manual(values = COND_COLORS) +
    labs(title = metrics_labels[metric], subtitle = subtitle,
         x = NULL, y = metrics_labels[metric]) +
    theme_pubr() +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", size = 11),
          plot.subtitle = element_text(size = 9, color = "gray30"))

  if (nrow(sig_pairs) > 0) {
    pair_lookup <- list("sil vs tem" = c("silence", "tempo_random"),
                        "sil vs bea" = c("silence", "beatmove_adaptatif"),
                        "tem vs bea" = c("tempo_random", "beatmove_adaptatif"))
    comparisons <- lapply(sig_pairs$pair, function(p) pair_lookup[[p]])
    p <- p + stat_compare_means(comparisons = comparisons,
                                  method = "wilcox.test", paired = TRUE,
                                  p.adjust.method = "bonferroni",
                                  label = "p.signif")
  }

  return(p)
}

plots <- purrr::map(METRICS, make_plot)
combined <- ggarrange(plotlist = plots, nrow = 1, ncol = length(METRICS))
ggsave(file.path(OUT_FIGURES, "05_stats_boxplots.png"),
       combined, width = 22, height = 6, dpi = 150, bg = "white")

cat("\nDone.\n")
