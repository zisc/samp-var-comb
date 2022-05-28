library(probabilistic)
library(ggplot2)
library(colorspace)
library(grid)
library(gtable)
library(lemon)
library(tikzDevice)
source("elsivier.R")
source("palette.R")
source("plotting_options.R")

save.file <- "plot_SP500.RData"

if (file.exists(save.file)) {
  load(save.file)
} else {
  source(file.path("SP500", "SP500.R"), chdir = TRUE)
  
  decimal_places <- 4
  # tbl_round <- function(x) {
  #   multiplier <- 10^decimal_places
  #  round(multiplier*x)/multiplier
  # }
  # tbl_num <- function(x) {
  #   if (tbl_round(x) == 0) { return("0") }
  #   sub("\\.$", "", sub("0+$", "", sprintf(paste0("%.", decimal_places, "f"), x)))
  # }
  tbl_round <- function(x) { x }
  tbl_num <- function(x) {
    sprintf(paste0("%#.", decimal_places, "g"), x)
  }
  
  sampling_dist_trim <- 0.0
  
  if (!dir.exists("figure")) {
    dir.create("figure")
  }
  
  plot_num_begin <- 5
  tbl_num_begin <- 1
  plot_file_name_stage <- NULL
  plot_file_name_opt <- NULL
  tbl_file_name <- NULL
  for (i in 1:length(periods)) {
    plot_file_name_stage[[ periods[i] ]] <- paste0("FIG", plot_num_begin+2*(i-1))
    plot_file_name_opt[[ periods[i] ]] <- paste0("FIG", plot_num_begin+2*(i-1)+1)
    tbl_file_name[[ periods[i] ]] <- paste0("TBL", tbl_num_begin+(i-1))
  }
  
  periods_rename <- list(Overall = "Overall period: January 3rd, 2017 to December 31st, 2021", Extreme = "Extreme period: January 2nd, 2020 to June 30th, 2020")
  performance_measures_plot <- c("LS", "CS10", "CS20")
  performance_measures_rename <- list(LS = "LS", CS5 = "$\\mathrm{CS}_{<5\\%}$", CS10 = "$\\mathrm{CS}_{<10\\%}$", CS20 = "$\\mathrm{CS}_{<20\\%}$")
  score_names_plot <- performance_measures_plot
  score_names_rename <- performance_measures_rename
  length_plotframes <- length(estimators)*length(score_names_plot)*length(performance_measures_plot)*monte_carlo_sample_size
  for (period in periods) {
    plot_frame <- tibble(
      Estimator = rep(NA_character_, length_plotframes),
      Optimize = rep(NA_character_, length_plotframes),
      Measure = rep(NA_character_, length_plotframes),
      Draw = rep(NA_real_, length_plotframes)
    )
    i <- 0
    for (measure in performance_measures_plot) {
      for (score_name in score_names_plot) {
        for (estimator in estimators) {
          draws <- results_table[[period]][[measure]][[score_name]][[estimator]]$`avg_draws`
          idx <- (i+1):(i+monte_carlo_sample_size)
          plot_frame$`Estimator`[idx] <- rep(estimator, monte_carlo_sample_size)
          plot_frame$`Optimize`[idx] <- rep(score_names_rename[[score_name]], monte_carlo_sample_size)
          plot_frame$`Measure`[idx] <- rep(performance_measures_rename[[measure]], monte_carlo_sample_size)
          plot_frame$`Draw`[idx] <- draws
          i <- i+monte_carlo_sample_size
        }
      }
    }
    plot_frame <- plot_frame %>%
      mutate(
        Optimize = paste("Optimize", Optimize),
        Measure = paste("Measure", Measure)
      ) %>%
      mutate(
        Optimize = factor(Optimize, levels = unique(Optimize)),
        Measure = factor(Measure, levels = unique(Measure))
      )
    plot_labels_stage <- plot_frame %>%
      select(Optimize, Measure) %>%
      distinct() %>%
      arrange(Optimize, Measure) %>%
      add_column(labels = LETTERS[1:(length(score_names_plot)*length(performance_measures_plot))])
    plot_labels_opt <- plot_frame %>%
      select(Estimator, Measure) %>%
      distinct() %>%
      arrange(Estimator, Measure) %>%
      add_column(labels = LETTERS[1:(length(estimators)*length(performance_measures_plot))])
    
    density_grid_stage <- ggplot(plot_frame, aes(Draw, colour = Estimator)) +
      geom_density(aes(y = ..scaled..), size = 0.7/.pt, trim = TRUE) +
      facet_rep_grid(
      # facet_grid(
        rows = vars(Optimize),
        cols = vars(Measure),
        scales = "free_x",
        switch = "y",
        repeat.tick.labels = TRUE
      ) +
      ggtitle(periods_rename[[period]]) +
      xlab("Out-of-Sample Score") +
      ylab("") +
      # scale_linetype_manual(values = score_lines) +
      scale_colour_manual(values = score_colour[c("One-Stage", "Two-Stage")], breaks = score_breaks) +
      theme_elsivier() +
      theme(
        legend.title = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        strip.text.y = element_text(angle = 90, margin = margin(r = 0.75, unit = "mm"))#,
        # panel.spacing.x = unit(4, "mm"),
        # panel.spacing.y = unit(4, "mm")
      )
    dgs_panels <- ggplot_build(density_grid_stage)$layout$panel_params
    density_grid_stage <- density_grid_stage +
      geom_text(
        aes(x = x, y = y, label = labels),
        plot_labels_stage %>%
          add_column(
            x = sapply(dgs_panels, function(p) { p$x.range[1] + letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
            y = sapply(dgs_panels, function(p) { p$y.range[2] - letter_label_y_offset*(p$y.range[2] - p$y.range[1]) })
          ),
        size = theme_elsivier_axis_title_size/.pt,
        inherit.aes = FALSE
      )
    tikz(
      file = file.path("figure", paste0(plot_file_name_stage[[period]], ".tex")),
      width = page_height_inches,
      height = full_height_width_inches_landscape,
      standAlone = TRUE
    )
    # See page_dimensions.tex.
    grid.newpage()
    print(density_grid_stage)
    dev.off()
    
    density_grid_opt <- ggplot(
      plot_frame %>%
        mutate(Optimize = gsub("Optimize ", "", Optimize)) %>%
        mutate(Optimize = factor(Optimize, levels = unique(Optimize))),
      aes(Draw, colour = Optimize)
    ) +
      geom_density(aes(y = ..scaled..), size = 0.7/.pt, trim = TRUE) +
      facet_rep_grid(
        rows = vars(Estimator),
        cols = vars(Measure),
        scales = "free_x",
        switch = "y",
        repeat.tick.labels = TRUE
      ) +
      ggtitle(periods_rename[[period]]) +
      xlab("Out-of-Sample Score") +
      ylab("") +
      # scale_linetype_manual(values = score_lines) +
      scale_colour_manual(values = score_colour[c("LS", "$\\mathrm{CS}_{<10\\%}$", "$\\mathrm{CS}_{<20\\%}$")], breaks = score_breaks) +
      theme_elsivier() +
      theme(
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        strip.text.y = element_text(angle = 90, margin = margin(r = 0.75, unit = "mm"))#,
        # panel.spacing.x = unit(4, "mm"),
        # panel.spacing.y = unit(4, "mm")
      )
    dgo_panels <- ggplot_build(density_grid_opt)$layout$panel_params
    density_grid_opt <- density_grid_opt +
      geom_text(
        aes(x = x, y = y, label = labels),
        plot_labels_opt %>%
          add_column(
            x = sapply(dgo_panels, function(p) { p$x.range[1] + 0.9*letter_label_x_offset*(p$x.range[2] - p$x.range[1]) }),
            y = sapply(dgo_panels, function(p) { p$y.range[2] - (2/3)*letter_label_y_offset*(p$y.range[2] - p$y.range[1]) })
          ),
        size = theme_elsivier_axis_title_size/.pt,
        inherit.aes = FALSE
      )
    tikz(
      file = file.path("figure", paste0(plot_file_name_opt[[period]], ".tex")),
      width = page_height_inches,
      height = full_height_width_inches_landscape,
      standAlone = TRUE
    )
    # See page_dimensions.tex
    grid.newpage()
    print(density_grid_opt)
    dev.off()
  }
  
  latex_cells <- list()
  for (period in periods) {
    latex_cells[[period]] <- list()
    for (test_score_name in score_names_plot) {
      latex_cells[[period]][[test_score_name]] <- list()
      col <- results_table[[period]][[test_score_name]]
      bold_avg <- list(score = score_names_plot[1], estimator = estimators[1])
      bold_ci <- list(score = score_names_plot[1], estimator = estimators[1])
      for (train_score_name in score_names_plot) {
        latex_cells[[period]][[test_score_name]][[train_score_name]] <- list()
        for (estimator in estimators) {
          result <- col[[train_score_name]][[estimator]]
          latex_cells[[period]][[test_score_name]][[train_score_name]][[estimator]] <- list(
            "Average" = tbl_num(result$Average),
            "95\\% CI" = paste0("(", tbl_num(result$CILow), ", ", tbl_num(result$CIHigh), ")")
          )
          if (result$Average > col[[bold_avg$score]][[bold_avg$estimator]]$Average) {
            bold_avg$score <- train_score_name
            bold_avg$estimator <- estimator
          }
          if ((result$CIHigh - result$CILow) < (col[[bold_ci$score]][[bold_ci$estimator]]$CIHigh - col[[bold_ci$score]][[bold_ci$estimator]]$CILow)) {
            bold_ci$score <- train_score_name
            bold_ci$estimator <- estimator
          }
        }
      }
      latex_cells[[period]][[test_score_name]][[bold_avg$score]][[bold_avg$estimator]]$Average <- paste0(
        "\\textbf{", latex_cells[[period]][[test_score_name]][[bold_avg$score]][[bold_avg$estimator]]$Average, "}"
      )
      latex_cells[[period]][[test_score_name]][[bold_ci$score]][[bold_ci$estimator]][["95\\% CI"]] <- paste0(
        "\\textbf{", latex_cells[[period]][[test_score_name]][[bold_ci$score]][[bold_ci$estimator]][["95\\% CI"]], "}"
      )
    }
  }
  
  latex_data_row <- function(period, train_score_name, estimator, stat) {
    row <- ""
    for (test_score_name in score_names_plot){
      row <- paste0(row, "& ", latex_cells[[period]][[test_score_name]][[train_score_name]][[estimator]][[stat]], " ")
    }
    row <- paste0(row, "\\\\ \n")
    return(row)
  }
  
  num_data_cols <- length(score_names_plot)
  
  table_page <- function(periods) {
    page <- paste0(
      "\\begin{tabular}{", paste0(rep("l", 4 + num_data_cols), collapse = ""), "}\n"
    )
    for (p in 1:length(periods)) {
      period <- periods[p]

      page <- paste0(
        page,
        "  \\rowcolor{Gray} \\multicolumn{", 4 + num_data_cols, "}{c}{Panel ", LETTERS[p], " -- ", periods_rename[[period]], "} \\\\\n",
        # "  \\\\\n",
        "  & & & & \\multicolumn{", num_data_cols, "}{c}{Out-of-Sample Score} \\\\\n",
        "  \\cmidrule(lr){", 5, "-", 4 + num_data_cols, "}\n",
        "  & & & "
      )
      for (score_name in score_names_plot) {
        page <- paste0(page, "& ", score_names_rename[[score_name]], " ")
      }
      page <- paste0(
        page,
        "\\\\\n",
        "  \\hline\n"
      )
      
      for (i in 1:length(score_names_plot)) {
        train_score_name <- score_names_plot[i]
        train_score_latex_name <- performance_measures_rename[[train_score_name]]
        page <- paste0(page, "  ")
        if (i == 1) {
          page <- paste0(
            page,
            "\\multirow{", 4*length(score_names_plot), "}{*}{\\rotatebox[origin=c]{90}{In-Sample Score}} "
            # "\\multirow{", 7*length(score_names_plot)-2, "}{*}{\\rotatebox[origin=c]{90}{In-Sample Score}} "#,
            # "\\multirow{", 2*length(performance_measures)-1, "}{*}{\\rotatebox[origin=c]{90}{Evaluated Out-of-Sample}} "
          )
        } # else {
          # page <- paste0(page, "\\\\\n  ")
        # }
        page <- paste0(page, "& ", train_score_latex_name)
        for (j in 1:length(estimators)) {
          estimator <- estimators[j]
          if (j != 1) { page <- paste0(page, "  &") }
          page <- paste0(page, " & ", estimator,
                         " & Average ",
                         latex_data_row(period, train_score_name, estimator, "Average"),
                         "  & & & 95\\% CI ",
                         latex_data_row(period, train_score_name, estimator, "95\\% CI")
          )
          if (i == length(score_names_plot) && j == length(estimators)) {
            page <- paste0(page, "  \\hline\n")
          } # else {
            # page <- paste0(page, "  \\\\\n")
          # }
        }
      }
      
      # if (p != length(periods)) {
      #   page <- paste0(
      #     page,
      #     "  \\\\\n",
      #     "  \\\\\n"
      #   )
      # }
    }
    
    page <- paste0(page, "\\end{tabular}\n")
    
    return(page)
  }
  
  plot_page <- function(period) {
    page <- paste0(
      "\\begin{sidewaysfigure}[!ht]\n",
      "\\includegraphics[width=\\textwidth]{", file.path("figure", paste0(plot_file_name_stage[[period]], ".pdf")), "}\n",
      "\\end{sidewaysfigure}\n",
      "\\newpage\n",
      "\\begin{sidewaysfigure}[!ht]\n",
      "\\includegraphics[width=\\textwidth]{", file.path("figure", paste0(plot_file_name_opt[[period]], ".pdf")), "}\n",
      "\\end{sidewaysfigure}\n"
    )
    return(page)
  }
  
  # for (period in periods) {
  #   cat(table_page(period), file = file.path("figure", paste0(tbl_file_name[[period]], ".tex")))
  # }
  # 
  # latex_document <- paste0(
  #   "\\documentclass[10pt]{article}\n",
  #   "\\usepackage[margin=2cm, landscape]{geometry}\n",
  #   "\\usepackage[T1]{fontenc}\n",
  #   "\\usepackage{booktabs, multirow, array, graphicx}\n",
  #   "\\begin{document}\n",
  #   "\\title{S&P 500 Forecast Performance}\n",
  #   "\n"
  # )
  # latex_document <- paste0(
  #   latex_document,
  #   "\\input{", file.path("figure", paste0(tbl_file_name[[ periods[1] ]], ".tex")), "}\n",
  #   "\n\\newpage\n\n",
  #   plot_page(periods[1])
  # )
  # for (i in 2:length(periods)) {
  #   latex_document <- paste0(
  #     latex_document,
  #     "\n\\newpage\n\n",
  #     "\\input{", file.path("figure", paste0(tbl_file_name[[ periods[i] ]], ".tex")), "}\n",
  #     "\n\\newpage\n\n",
  #     plot_page(periods[i])
  #   )
  # }
  # latex_document <- paste0(latex_document, "\\end{document}\n")
  
  cat(table_page(periods), file = file.path("figure", paste0("TBL", tbl_num_begin, ".tex")))
  
  latex_document <- paste0(
    "\\documentclass[10pt]{article}\n",
    "\\usepackage[margin=2cm]{geometry}\n",
    "\\usepackage[T1]{fontenc}\n",
    "\\usepackage{booktabs, multirow, array, graphicx, rotating, color, colortbl}\n",
    "\\definecolor{Gray}{gray}{0.9}\n",
    "\\begin{document}\n",
    "\\title{S&P 500 Forecast Performance}\n",
    "\n",
    "\\begin{table}[!h]\n",
    "\\centering\n",
    "\\input{", file.path("figure", paste0("TBL", tbl_num_begin, ".tex")), "}\n",
    "\\end{table}\n"
  )
  for (period in periods) {
    latex_document <- paste0(
      latex_document,
      "\n\\newpage\n\n",
      plot_page(period)
    )
  }
  latex_document <- paste0(latex_document, "\\end{document}\n")
  
  cat(latex_document, file = "plot_SP500.tex")
  
  save(latex_document, file = save.file)
  
}
