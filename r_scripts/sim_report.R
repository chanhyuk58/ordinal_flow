library(here)
library(ggplot2)
library(dplyr)
library(tidyr)

summary_path <- here("mc_summary", "mc_summary.csv")
if (!file.exists(summary_path)) {
  stop(paste("Aggregated summary file not found at:", summary_path))
}
df <- read.csv(summary_path)

# Ensure proper types
df$n <- as.integer(df$n)
df$index <- as.integer(df$index)
df$estimate_mean <- as.numeric(df$estimate_mean)
df$bias <- as.numeric(df$bias)
df$rmse <- as.numeric(df$rmse)
df$truth <- as.numeric(df$truth)

# Colorblind safe colors
okabe_ito <- c(
  "empirical"        = "#999999",  # Slate Gray
  "ordered_probit"   = "#E69F00",  # Orange
  "ordered_logit"    = "#56B4E9",  # Sky Blue
  "structured_flow"  = "#009E73",  # Bluish Green
  "model_free_flow"  = "#CC79A7"   # Reddish Purple
)

# Labels for plotting
model_labels <- c(
  "empirical"        = "Empirical",
  "ordered_probit"   = "Ordered Probit",
  "ordered_logit"    = "Ordered Logit",
  "structured_flow"  = "Structured Flow",
  "model_free_flow"  = "Model-Free Flow"
)

# ---------------------------------------------------------------------
# Part 1: Main Text Figures (Separate plots per setting, N=1000)
# ---------------------------------------------------------------------

plot_setting_effects <- function(setting_name, output_filename) {
  plot_df <- df %>%
    filter(
      setting == setting_name,
      n == 2000,
      metric %in% c("category_effect", "cum_ge_effect")
    ) %>%
    mutate(
      ymin = estimate_mean - 1.96 * (estimate_sd / sqrt(n_rep)),
      ymax = estimate_mean + 1.96 * (estimate_sd / sqrt(n_rep)),
      model = factor(model, levels = names(okabe_ito)),
      facet_label = factor(
        ifelse(metric == "category_effect", paste("Category", index), paste("P(Y >=", index, ")")),
        levels = c(paste("Category", 1:5), paste("P(Y >=", 2:5))
      )
    )
  
  # Ground truth horizontal lines
  truth_df <- plot_df %>%
    group_by(facet_label) %>%
    summarise(truth = unique(truth), .groups = 'drop')

  p <- ggplot(plot_df, aes(x = model, y = estimate_mean, color = model)) +
    # Thin dotted line for 0 baseline
    geom_hline(yintercept = 0, linetype = "dotted", color = "gray60", linewidth = 0.5) +
    
    # Bold solid line for Ground Truth
    geom_hline(data = truth_df, aes(yintercept = truth), color = "black", linewidth = 0.8, linetype = "solid") +
    
    # Dodged pointranges to prevent any visual overlap
    geom_pointrange(aes(ymin = ymin, ymax = ymax), position = position_dodge(width = 0.6), linewidth = 0.8, size = 0.6) +
    facet_wrap(~ facet_label, scales = "free_y", nrow = 1) +
    scale_color_manual(values = okabe_ito, labels = model_labels) +
    scale_x_discrete(labels = NULL, breaks = NULL) +  # Hide x-axis labels since legend describes them
    labs(
      x = NULL,
      y = "Treatment Effect"
    ) +
    theme_classic(base_size = 12) +
    theme(
      strip.background = element_rect(fill = "#F5F5F2", color = NA),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.grid.major.y = element_line(color = "#EAEAEA", linewidth = 0.3)
    )
  
  ggsave(here("figures", output_filename), plot = p, width = 15, height = 4, device = "pdf")
}

# Generate Main Text Figures
print("Generating Main Text Figures...")
plot_setting_effects("normal_linear", "fig_app_normal_linear_N2000.pdf")
plot_setting_effects("heteroskedastic", "fig_heteroskedastic_N2000.pdf")
plot_setting_effects("polarized_mixture", "fig_polarized_mixture_N2000.pdf")

# ---------------------------------------------------------------------
# Part 2: Main Text Wasserstein RMSE Convergence Plot
# ---------------------------------------------------------------------
print("Generating Wasserstein Convergence Plot...")
wasserstein_df <- df %>%
  filter(
    setting %in% c("normal_linear", "heteroskedastic", "polarized_mixture"),
    metric == "wasserstein_unit"
  ) %>%
  mutate(
    model = factor(model, levels = names(okabe_ito)),
    setting_label = factor(
      case_when(
        setting == "normal_linear" ~ "Normal Linear",
        setting == "heteroskedastic" ~ "Heteroskedastic",
        setting == "polarized_mixture" ~ "Polarized Mixture"
      ),
      levels = c("Normal Linear", "Heteroskedastic", "Polarized Mixture")
    )
  )

p_conv <- ggplot(wasserstein_df, aes(x = factor(n), y = rmse, group = model, color = model)) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.5) +
  facet_wrap(~ setting_label, scales = "free_y") +
  scale_color_manual(values = okabe_ito, labels = model_labels) +
  labs(
    x = "Sample Size (N)",
    y = "Wasserstein Distance RMSE"
  ) +
  theme_classic(base_size = 12) +
  theme(
    strip.background = element_rect(fill = "#F5F5F2", color = NA),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.grid.major.y = element_line(color = "#EAEAEA", linewidth = 0.3)
  )
ggsave(here("figures", "fig_wasserstein_convergence.pdf"), plot = p_conv, width = 10, height = 4, device = "pdf")

# ---------------------------------------------------------------------
# Part 3: Appendix Figures (Other Settings and Sample Sizes)
# ---------------------------------------------------------------------
print("Generating Appendix Figures...")
plot_setting_effects("logistic_linear", "fig_logistic_linear_N2000.pdf")
plot_setting_effects("skewed_lognormal", "fig_app_skewed_lognormal_N2000.pdf")
plot_setting_effects("nonlinear_moderates", "fig_app_nonlinear_moderates_N2000.pdf")
plot_setting_effects("high_dimensional", "fig_app_high_dimensional_N2000.pdf")

# ---------------------------------------------------------------------
# Part 4: Main Text LaTeX Table (Polished booktabs)
# ---------------------------------------------------------------------
print("Generating Polished Main LaTeX Table...")
table_df <- df %>%
  filter(
    setting %in% c("normal_linear", "heteroskedastic", "polarized_mixture"),
    n == 1000,
    metric == "wasserstein_unit"
  ) %>%
  arrange(setting, match(model, names(okabe_ito)))

# Construct LaTeX table string manually to bypass standard formatting limitations
latex_output <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Causal Estimation Diagnostics (Wasserstein Unit, $N=1000$)}",
  "\\label{tab:main_diagnostics}",
  "\\begin{tabular}{llcccc}",
  "\\toprule",
  "\\textbf{DGP Setting} & \\textbf{Estimator} & \\textbf{True Value} & \\textbf{MC Mean} & \\textbf{Bias} & \\textbf{RMSE} \\\\",
  "\\midrule"
)

current_setting <- ""
for (i in 1:nrow(table_df)) {
  row <- table_df[i, ]
  
  # Add visual break line between settings
  if (row$setting != current_setting) {
    if (current_setting != "") latex_output <- c(latex_output, "\\midrule")
    current_setting <- row$setting
    setting_print <- case_when(
      current_setting == "normal_linear" ~ "\\textbf{Normal Linear}",
      current_setting == "heteroskedastic" ~ "\\textbf{Heteroskedastic}",
      current_setting == "polarized_mixture" ~ "\\textbf{Polarized Mixture}"
    )
  } else {
    setting_print <- ""
  }
  
  model_print <- model_labels[row$model]
  
  # Highlight flow wins in bold font
  if (row$model %in% c("structured_flow", "model_free_flow") && row$setting != "normal_linear") {
    model_print <- paste0("\\textbf{", model_print, "}")
    bias_print <- paste0("\\textbf{", sprintf("%.5f", row$bias), "}")
    rmse_print <- paste0("\\textbf{", sprintf("%.4f", row$rmse), "}")
  } else {
    bias_print <- sprintf("%.5f", row$bias)
    rmse_print <- sprintf("%.4f", row$rmse)
  }
  
  latex_output <- c(
    latex_output,
    sprintf("%s & %s & %.4f & %.4f & %s & %s \\\\",
            setting_print, model_print, row$truth, row$estimate_mean, bias_print, rmse_print)
  )
}

latex_output <- c(
  latex_output,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(latex_output, here("tables", "table_main_diagnostics.tex"))

# ---------------------------------------------------------------------
# Part 5: Complete Appendix LaTeX Table (Polished booktabs)
# ---------------------------------------------------------------------
print("Generating Exhaustive Appendix LaTeX Table...")

app_table_df <- df %>%
  filter(metric == "wasserstein_unit") %>%
  arrange(setting, n, match(model, names(okabe_ito)))

app_latex_output <- c(
  "\\begin{table}[p]",
  "\\centering",
  "\\small",
  "\\caption{Full Simulation Results: Wasserstein Distance to Truth Across All Settings}",
  "\\label{tab:app_full_wasserstein}",
  "\\begin{tabular}{lllccccc}",
  "\\toprule",
  "\\textbf{DGP Setting} & \\textbf{N} & \\textbf{Estimator} & \\textbf{True} & \\textbf{Mean} & \\textbf{SD} & \\textbf{Bias} & \\textbf{RMSE} \\\\",
  "\\midrule"
)

current_setting <- ""
current_n <- -1
for (i in 1:nrow(app_table_df)) {
  row <- app_table_df[i, ]
  
  if (row$setting != current_setting) {
    if (current_setting != "") app_latex_output <- c(app_latex_output, "\\midrule")
    current_setting <- row$setting
    setting_print <- paste0("\\textbf{", gsub("_", " ", current_setting) %>% tools::toTitleCase(), "}")
    current_n <- -1
  } else {
    setting_print <- ""
  }
  
  if (row$n != current_n) {
    n_print <- as.character(row$n)
    current_n <- row$n
    if (setting_print == "" && i > 1 && app_table_df[i-1, ]$n != current_n) {
      app_latex_output <- c(app_latex_output, "\\cline{2-8}")
    }
  } else {
    n_print <- ""
  }
  
  model_print <- model_labels[row$model]
  
  app_latex_output <- c(
    app_latex_output,
    sprintf("%s & %s & %s & %.4f & %.4f & %.4f & %.5f & %.4f \\\\",
            setting_print, n_print, model_print, row$truth, row$estimate_mean, row$estimate_sd, row$bias, row$rmse)
  )
}

app_latex_output <- c(
  app_latex_output,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(app_latex_output, here("tables", "table_app_full_wasserstein.tex"))
print("Done! All figures and LaTeX tables generated successfully.")
