library(here)
library(ggplot2)
library(dplyr)
library(tidyr)

dir.create(here("figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("tables"), recursive = TRUE, showWarnings = FALSE)

# Okabe-Ito color palette for Colorblind safety
okabe_ito <- c(
  "empirical"        = "#999999",  # Gray
  "oprobit"          = "#E69F00",  # Orange
  "ologit"           = "#56B4E9",  # Sky Blue
  "structured flow"  = "#009E73",  # Bluish Green
  "model_free flow"  = "#CC79A7"   # Reddish Purple
)

model_labels <- c(
  "empirical"        = "Empirical",
  "oprobit"          = "Ordered Probit",
  "ologit"           = "Ordered Logit",
  "structured flow"  = "Structured Flow",
  "model_free flow"  = "Model-Free Flow"
)

# =====================================================================
# Helper: LaTeX Table Generator
# =====================================================================
export_latex_table <- function(data_path, output_filename, J, title_label) {
  df_raw <- read.csv(data_path)
  
  # Format cells as Estimate (Std. Error)
  df_wide <- df_raw %>%
    mutate(
      formatted = sprintf("%.3f (%.3f)", estimate, se),
      model = factor(model, levels = names(okabe_ito))
    ) %>%
    select(model, metric, index, formatted) %>%
    pivot_wider(names_from = model, values_from = formatted)
  
  # Group and order rows logically
  rows_wasserstein <- df_wide %>% filter(metric == "wasserstein_unit")
  rows_category <- df_wide %>% filter(metric == "category_effect") %>% arrange(index)
  rows_cum_ge <- df_wide %>% filter(metric == "cum_ge_effect") %>% arrange(index)
  
  latex <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    sprintf("\\caption{Empirical Application Estimates: %s}", title_label),
    sprintf("\\label{tab:%s}", gsub(".tex", "", output_filename)),
    "\\small",
    "\\begin{tabular}{lccccc}",
    "\\toprule",
    "\\textbf{Metric / Category} & \\textbf{Empirical} & \\textbf{Ordered Probit} & \\textbf{Ordered Logit} & \\textbf{Structured Flow} & \\textbf{Model-Free Flow} \\\\",
    "\\midrule",
    # 1. Wasserstein
    sprintf("Wasserstein Unit & %s & %s & %s & %s & %s \\\\",
            rows_wasserstein$empirical, rows_wasserstein$oprobit, rows_wasserstein$ologit, 
            rows_wasserstein$`structured flow`, rows_wasserstein$`model_free flow`),
    "\\midrule",
    "\\multicolumn{6}{l}{\\textit{Category Effects (AME)}} \\\\"
  )
  
  # 2. Category Effects
  for (i in 1:nrow(rows_category)) {
    row <- rows_category[i, ]
    latex <- c(latex, sprintf("  Category %d & %s & %s & %s & %s & %s \\\\",
                              row$index, row$empirical, row$oprobit, row$ologit, 
                              row$`structured flow`, row$`model_free flow`))
  }
  
  # 3. Cumulative GE Effects
  latex <- c(latex, "\\midrule", "\\multicolumn{6}{l}{\\textit{Cumulative Generalization Effects (CGE)}} \\\\")
  for (i in 1:nrow(rows_cum_ge)) {
    row <- rows_cum_ge[i, ]
    latex <- c(latex, sprintf("  P(Y $\\ge$ %d) & %s & %s & %s & %s & %s \\\\",
                              row$index, row$empirical, row$oprobit, row$ologit, 
                              row$`structured flow`, row$`model_free flow`))
  }
  
  latex <- c(latex, "\\bottomrule", "\\end{tabular}", "\\end{table}")
  
  writeLines(latex, here("tables", output_filename))
}

# =====================================================================
# Helper: Pointrange Plot Generator
# =====================================================================

export_empirical_figure <- function(data_path, output_filename, index_label = NA) {
  df_raw <- read.csv(data_path)
  
  df_plot <- df_raw %>%
    filter(metric == "category_effect")
  
  if (all(is.na(index_label))) {
    # Fallback to default Category 1, Category 2, etc.
    df_plot <- df_plot %>%
      mutate(index_label = factor(paste("Category", index), 
                                  levels = paste("Category", sort(unique(index)))))
  } else {
    # Map the numerical index directly to your custom character vector
    df_plot <- df_plot %>%
      mutate(index_label = factor(index_label[index], levels = index_label))
  }
  
  df_plot <- df_plot %>%
    mutate(
      ymin = estimate - 1.96 * se,
      ymax = estimate + 1.96 * se,
      model = factor(model, levels = names(okabe_ito))
    )
  
  p <- ggplot(df_plot, aes(x = model, y = estimate, color = model)) +
    # Thin dotted line for 0 baseline
    geom_hline(yintercept = 0, linetype = "dotted", color = "gray60", linewidth = 0.5) +
    
    # Dodged pointranges with slight offsets to prevent overlapping error bars
    geom_pointrange(aes(ymin = ymin, ymax = ymax), position = position_dodge(width = 0.5), linewidth = 0.8, size = 0.6) +
    facet_wrap(~ index_label, nrow = 1) +
    scale_color_manual(values = okabe_ito, labels = model_labels) +
    scale_x_discrete(labels = NULL, breaks = NULL) +  # Hide x axis labels since legend describes them
    labs(
      x = NULL,
      y = "Estimated Treatment Effect"
    ) +
    theme_classic(base_size = 12) +
    theme(
      strip.background = element_rect(fill = "#F5F5F2", color = NA),
      strip.text = element_text(face = "bold", size = 10),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.grid.major.y = element_line(color = "#EAEAEA", linewidth = 0.3)
    )
  
  ggsave(here("figures", output_filename), plot = p, width = 11, height = 4, device = "pdf")
}

# =====================================================================
# Run Processors
# =====================================================================
print("Processing Tomz (2020) outputs...")
export_latex_table(here("replication_results", "tomz_results.csv"), "table_tomz_empirical.tex", J = 5, "Support for Strike (Tomz 2020)")
tomz_index = c("Strongly Disagree", "Disagree", "Neigher", "Agree", "Strongly Agree")
export_empirical_figure(here("replication_results", "tomz_results.csv"), "fig_tomz_empirical.pdf", tomz_index)

print("\nProcessing Mattingly (2023a) outputs...")
export_latex_table(here("replication_results", "mattingly_results.csv"), "table_mattingly_empirical.tex", J = 6, "Support for Political Model (Mattingly 2023a)")
mattingly_index = c("Strongly Prefer US", "Prefer US", "Somewhat Prefer US", "Somewhat Prefer China", "Prefer China", "Strongly Prefer China")
export_empirical_figure(here("replication_results", "mattingly_results.csv"), "fig_mattingly_empirical.pdf", mattingly_index)

print("\nAll figures and LaTeX tables successfully written to 'figures/' and 'tables/' directories.")
