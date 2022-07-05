library(ggplot2)
library(ggrepel)
library(egg)
library(grid)
library(gridExtra)
library(lemon)
library(latex2exp)
library(tikzDevice)
library(tsibble)
library(probabilistic)
source("elsivier.R")
source("palette.R")
source("plotting_options.R")
source(file.path("simulation", "momentify.R"))

group_by <- dplyr::group_by
mutate <- dplyr::mutate
filter <- dplyr::filter
select <- dplyr::select
distinct <- dplyr::distinct
arrange <- dplyr::arrange
add_column <- tibble::add_column
summarise <- dplyr::summarise
full_join <- dplyr::full_join

simulation_plots_file <- "plot_simulation.RData"

if (file.exists(simulation_plots_file)) {
  load(simulation_plots_file)
} else {
  simulation_results_file <<- file.path("simulation", "simulation.RData")
  if (!file.exists(simulation_results_file)) {
    source(file.path("simulation", "simulation.R"), local = new.env(), chdir = TRUE)
  }
  load(simulation_results_file)
  
  exp_sample_size_min <- 500
  var_sample_size_min <- 500
  
  simulation_results_moments <- simulation_results$moments %>%
    rename(Optimize = Optimise) %>%
    mutate(
      Estimator = gsub("Optimise", "Optimize", Estimator),
      Optimize = gsub("Optimise", "Optimize", Optimize)
    ) %>%
    mutate(
      Optimize = gsub("Censored Log Score", "$\\\\mathrm{CS}_{<20\\\\%}$", Optimize),
      Measure = gsub("Censored Log Score", "$\\\\mathrm{CS}_{<20\\\\%}$", Measure)
    ) %>%
    mutate(
      Optimize = gsub("Log Score", "LS", Optimize),
      Measure = gsub("Log Score", "LS", Measure)
    ) %>%
    mutate(
      Optimize = factor(Optimize, levels = c("Optimize LS", "Optimize $\\mathrm{CS}_{<20\\%}$")),
      Measure = factor(Measure, levels = c("Measure LS", "Measure $\\mathrm{CS}_{<20\\%}$"))
    ) %>%
    group_by(Estimator) %>%
    mutate(`Estimator ID` = cur_group_id()) %>%
    group_by(Optimize) %>%
    mutate(`Optimize ID` = cur_group_id()) %>%
    group_by(`Sample Size`) %>%
    mutate(`Sample Size ID` = cur_group_id()) %>%
    group_by()
  
  simulation_results_exp <- filter(simulation_results_moments, `Sample Size` >= exp_sample_size_min)
  simulation_results_var <- filter(simulation_results_moments, `Sample Size` >= var_sample_size_min)
  
  plot_labels_stage <- simulation_results_moments %>%
    select(Optimize, Measure) %>%
    distinct() %>%
    arrange(Measure, Optimize) %>%
    add_column(labels_stage = c("A", "B", "C", "D"))
  
  plot_labels_opt <- simulation_results_moments %>%
    select(Estimator, Measure) %>%
    distinct() %>%
    filter(Estimator %in% c("One-Stage", "Two-Stage")) %>%
    arrange(Measure, Estimator) %>%
    add_column(labels_opt = plot_labels_stage$labels_stage)
  
  divergence_expectation_plot_stage <<- ggplot(
    simulation_results_exp,
    aes(x = `Sample Size`, colour = Estimator)
  ) +
    geom_line(aes(y = `Average Divergence`), size = 0.6/.pt) +
    geom_line(aes(y = `Expected Divergence CI Low`, group = Estimator, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    geom_line(aes(y = `Expected Divergence CI High`, group = Estimator, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    facet_rep_grid(
      rows = vars(`Measure`),
      cols = vars(`Optimize`),
      scales = "free_y",
      repeat.tick.labels = TRUE
    ) +
    scale_y_continuous(trans = "log10") +
    ylab("Expected Divergence (Lower is Better)") +
    scale_colour_manual(name = NULL, values = score_colour[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_linetype_manual(name = NULL, values = score_lines["95\\% Confidence Interval"], breaks = score_breaks) +
    scale_shape_manual(name = NULL, values = score_shapes[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_fill_manual(name = NULL, values = score_fill, breaks = score_breaks) +
    guides(shape = guide_legend(order = 1), colour = guide_legend(order = 2), linetype = guide_legend(order = 3)) +
    theme_elsivier()
  deps_panels <- ggplot_build(divergence_expectation_plot_stage)$layout$panel_params
  divergence_expectation_plot_stage <<- divergence_expectation_plot_stage +
    geom_text_repel(
      aes(x = x, y = y, label = labels_stage),
      plot_labels_stage %>%
        add_column(
          x = sapply(deps_panels, function(p) { p$x.range[1] + letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
          y = sapply(deps_panels, function(p) { 10^(p$y.range[2] + letter_label_y_offset*(p$y.range[2] - p$y.range[1])) })
        ),
      size = theme_elsivier_axis_title_size/ .pt,
      inherit.aes = FALSE
    )
  
  if (!dir.exists("figure")) {
    dir.create("figure")
  }
  
  tikz(
    file = file.path("figure", "FIG1.tex"),
    width = page_width_inches,
    height = full_width_height_inches,
    standAlone = TRUE
  )
  print(divergence_expectation_plot_stage)
  dev.off()
  
  divergence_expectation_plot_stage_presentation <<- ggplot(
    simulation_results_exp,
    aes(x = `Sample Size`)
  ) +
    geom_line(aes(y = `Average Divergence`, colour = Estimator), size = 0.6/.pt) +
    geom_ribbon(aes(ymin = `Expected Divergence CI Low`, ymax = `Expected Divergence CI High`, fill = Estimator), alpha = 0.2) +
    facet_rep_grid(
      rows = vars(`Measure`),
      cols = vars(`Optimize`),
      scales = "free_y",
      repeat.tick.labels = TRUE
    ) +
    scale_y_continuous(trans = "log10") +
    ylab("Expected Divergence (Lower is Better)") +
    scale_colour_manual(name = NULL, values = score_colour[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_linetype_manual(name = NULL, values = score_lines["95\\% Confidence Interval"], breaks = score_breaks) +
    scale_shape_manual(name = NULL, values = score_shapes[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_fill_manual(name = NULL, values = score_colour[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    guides(shape = guide_legend(order = 1), colour = guide_legend(order = 2), linetype = guide_legend(order = 3), fill = "none") +
    theme_elsivier()
  deps_panels <- ggplot_build(divergence_expectation_plot_stage)$layout$panel_params
  divergence_expectation_plot_stage <<- divergence_expectation_plot_stage +
    geom_text_repel(
      aes(x = x, y = y, label = labels_stage),
      plot_labels_stage %>%
        add_column(
          x = sapply(deps_panels, function(p) { p$x.range[1] + letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
          y = sapply(deps_panels, function(p) { 10^(p$y.range[2] + letter_label_y_offset*(p$y.range[2] - p$y.range[1])) })
        ),
      size = theme_elsivier_axis_title_size/ .pt,
      inherit.aes = FALSE
    )
  
  tikz(
    file = file.path("figure", "FIG1p.tex"),
    width = page_width_inches,
    height = full_width_height_inches,
    standAlone = TRUE
  )
  print(divergence_expectation_plot_stage_presentation)
  dev.off()
  
  divergence_expectation_plot_opt <<- ggplot(
    simulation_results_exp %>% filter(Estimator %in% c("One-Stage", "Two-Stage")) %>%
      mutate(Optimize = gsub("Optimize ", "", Optimize)) %>%
      mutate(Optimize = factor(Optimize, levels = unique(Optimize))),
    aes(x = `Sample Size`, colour = Optimize)
  ) +
    geom_line(aes(y = `Average Divergence`), size = 0.6/.pt) +
    geom_line(aes(y = `Expected Divergence CI Low`, group = Optimize, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    geom_line(aes(y = `Expected Divergence CI High`, group = Optimize, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    facet_rep_grid(
      rows = vars(`Measure`),
      cols = vars(`Estimator`),
      scale = "free_y",
      repeat.tick.labels = TRUE
    ) +
    scale_y_continuous(trans = "log10") +
    ylab("Expected Divergence (Lower is Better)") +
    scale_colour_manual(values = score_colour[c("LS", "$\\mathrm{CS}_{<20\\%}$")], breaks = score_breaks) +
    scale_shape_manual(values = score_shapes[c("LS", "$\\mathrm{CS}_{<20\\%}$")], breaks = score_breaks) +
    scale_linetype_manual(name = NULL, values = score_lines["95\\% Confidence Interval"], breaks = score_breaks) +
    scale_fill_manual(name = NULL, values = score_fill, breaks = score_breaks) +
    guides(shape = guide_legend(order = 1), colour = guide_legend(order = 2), linetype = guide_legend(order = 3)) +
    theme_elsivier()
  depo_panels <- ggplot_build(divergence_expectation_plot_opt)$layout$panel_params
  divergence_expectation_plot_opt <<- divergence_expectation_plot_opt +
    geom_text_repel(
      aes(x = x, y = y, label = labels_opt),
      plot_labels_opt %>%
        add_column(
          x = sapply(depo_panels, function(p) { p$x.range[1] + letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
          y = sapply(depo_panels, function(p) { 10^(p$y.range[2] + letter_label_y_offset*(p$y.range[2] - p$y.range[1])) })
        ),
      size = theme_elsivier_axis_title_size/.pt,
      inherit.aes = FALSE
    )
  
  tikz(
    file = file.path("figure", "FIG2.tex"),
    width = page_width_inches,
    height = full_width_height_inches,
    standAlone = TRUE
  )
  # See page_dimensions.tex.
  print(divergence_expectation_plot_opt)
  dev.off()
  
  score_variance_plot_stage <<- ggplot(
    simulation_results_var,
    aes(x = `Sample Size`, colour = Estimator)
  ) +
    geom_line(aes(y = `Sample Size`*Variance), size = 0.6/.pt) +
    geom_line(aes(y = `Sample Size`*`Variance CI Low`, group = Estimator, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    geom_line(aes(y = `Sample Size`*`Variance CI High`, group = Estimator, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    facet_rep_grid(
      rows = vars(`Measure`),
      cols = vars(`Optimize`),
      scale = "free_y",
      repeat.tick.labels = TRUE
    ) +
    scale_y_continuous(trans = "log10") +
    ylab("$\\text{Sample Size} \\times \\text{Var}\\left[\\mathcal{S}_{0}(\\hat{\\vartheta}_{n})\\right]$ (Lower is Better)") +
    scale_colour_manual(name = NULL, values = score_colour[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_linetype_manual(name = NULL, values = score_lines["95\\% Confidence Interval"], breaks = score_breaks) +
    scale_shape_manual(name = NULL, values = score_shapes[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_fill_manual(name = NULL, values = score_fill, breaks = score_breaks) +
    guides(shape = guide_legend(order = 1), colour = guide_legend(order = 2), linetype = guide_legend(order = 3)) +
    theme_elsivier() +
    theme(axis.title.y = element_text(margin = margin(r = 2, unit = "mm")))
  svps_panels = ggplot_build(score_variance_plot_stage)$layout$panel_params
  score_variance_plot_stage <<- score_variance_plot_stage +
    geom_text_repel(
      aes(x = x, y = y, label = labels_stage),
      plot_labels_stage %>%
        add_column(
          x = sapply(svps_panels, function(p) { p$x.range[1] + letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
          y = sapply(svps_panels, function(p) { 10^(p$y.range[2] + letter_label_y_offset*(p$y.range[2] - p$y.range[1])) })
        ),
      size = theme_elsivier_axis_title_size/.pt,
      inherit.aes = FALSE
    )
  
  tikz(
    file = file.path("figure", "FIG3.tex"),
    width = page_width_inches,
    height = full_width_height_inches,
    standAlone = TRUE
  )
  print(score_variance_plot_stage)
  dev.off()
  
  score_variance_plot_stage_presentation <<- ggplot(
    simulation_results_var,
    aes(x = `Sample Size`)
  ) +
    geom_line(aes(y = `Sample Size`*Variance, colour = Estimator), size = 0.6/.pt) +
    geom_ribbon(aes(ymin = `Sample Size`*`Variance CI Low`, ymax = `Sample Size`*`Variance CI High`, fill = Estimator), alpha = 0.2) +
    facet_rep_grid(
      rows = vars(`Measure`),
      cols = vars(`Optimize`),
      scale = "free_y",
      repeat.tick.labels = TRUE
    ) +
    scale_y_continuous(trans = "log10") +
    ylab("$\\text{Sample Size} \\times \\text{Var}\\left[\\mathcal{S}_{0}(\\hat{\\vartheta}_{n})\\right]$ (Lower is Better)") +
    scale_colour_manual(name = NULL, values = score_colour[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_linetype_manual(name = NULL, values = score_lines["95\\% Confidence Interval"], breaks = score_breaks) +
    scale_shape_manual(name = NULL, values = score_shapes[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    scale_fill_manual(name = NULL, values = score_colour[c("One-Stage", "Two-Stage", "Two-Stage - Weights Fixed at Limit Optimizer")], breaks = score_breaks) +
    guides(shape = guide_legend(order = 1), colour = guide_legend(order = 2), linetype = guide_legend(order = 3), fill = "none") +
    theme_elsivier() +
    theme(axis.title.y = element_text(margin = margin(r = 2, unit = "mm")))
  svps_panels = ggplot_build(score_variance_plot_stage)$layout$panel_params
  score_variance_plot_stage <<- score_variance_plot_stage +
    geom_text_repel(
      aes(x = x, y = y, label = labels_stage),
      plot_labels_stage %>%
        add_column(
          x = sapply(svps_panels, function(p) { p$x.range[1] + letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
          y = sapply(svps_panels, function(p) { 10^(p$y.range[2] + letter_label_y_offset*(p$y.range[2] - p$y.range[1])) })
        ),
      size = theme_elsivier_axis_title_size/.pt,
      inherit.aes = FALSE
    )
  
  tikz(
    file = file.path("figure", "FIG3p.tex"),
    width = page_width_inches,
    height = full_width_height_inches,
    standAlone = TRUE
  )
  print(score_variance_plot_stage_presentation)
  dev.off()
  
  score_variance_plot_opt <<- ggplot(
    simulation_results_var %>%
      filter(Estimator %in% c("One-Stage", "Two-Stage")) %>%
      mutate(Optimize = gsub("Optimize ", "", Optimize)) %>%
      mutate(Optimize = factor(Optimize, levels = unique(Optimize))),
    aes(x = `Sample Size`, colour = Optimize)
  ) +
    geom_line(aes(y = `Sample Size`*Variance), size = 0.6/.pt) +
    geom_line(aes(y = `Sample Size`*`Variance CI Low`, group = Optimize, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    geom_line(aes(y = `Sample Size`*`Variance CI High`, group = Optimize, linetype = "95\\% Confidence Interval"), size = 0.4/.pt) +
    facet_rep_grid(
      rows = vars(`Measure`),
      cols = vars(`Estimator`),
      scale = "free_y",
      repeat.tick.labels = TRUE
    ) +
    scale_y_continuous(trans = "log10") +
    ylab("$\\text{Sample Size} \\times \\text{Var}\\left[\\mathcal{S}_{0}(\\hat{\\vartheta}_{n})\\right]$ (Lower is Better)") +
    scale_colour_manual(values = score_colour[c("LS", "$\\mathrm{CS}_{<20\\%}$")], breaks = score_breaks) +
    scale_shape_manual(values = score_shapes[c("LS", "$\\mathrm{CS}_{<20\\%}$")], breaks = score_breaks) +
    scale_linetype_manual(name = NULL, values = score_lines["95\\% Confidence Interval"], breaks = score_breaks) +
    scale_fill_manual(name = NULL, values = score_fill, breaks = score_breaks) +
    guides(shape = guide_legend(order = 1), colour = guide_legend(order = 2), linetype = guide_legend(order = 3)) +
    theme_elsivier() +
    theme(axis.title.y = element_text(margin = margin(r = 2, unit = "mm")))
  svpo_panels = ggplot_build(score_variance_plot_opt)$layout$panel_params
  score_variance_plot_opt <<- score_variance_plot_opt +
    geom_text_repel(
      aes(x = x, y = y, label = labels_opt),
      plot_labels_opt %>%
        add_column(
          x = sapply(svpo_panels, function(p) { p$x.range[1] + letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
          y = sapply(svpo_panels, function(p) { 10^(p$y.range[2] + letter_label_y_offset*(p$y.range[2] - p$y.range[1])) })
        ),
      size = theme_elsivier_axis_title_size/.pt,
      inherit.aes = FALSE
    )
  
  tikz(
    file = file.path("figure", "FIG4.tex"),
    width = page_width_inches,
    height = full_width_height_inches,
    standAlone = TRUE
  )
  print(score_variance_plot_opt)
  dev.off()
  
  save(
    divergence_expectation_plot_stage,
    divergence_expectation_plot_opt,
    score_variance_plot_stage,
    score_variance_plot_opt,
    file = simulation_plots_file,
    compress = TRUE,
    compression_level = 9
  )
}
