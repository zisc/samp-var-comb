library(probabilistic)
library(readr)
library(tsibble)
library(openssl)

seed_file <- "seed.txt"
if (!file.exists(seed_file)) {
  write(as.integer(Sys.time()), file = seed_file)
}
seed <- scan(file = seed_file, what = integer(), quiet = TRUE)
set.seed(seed)
seed_torch_rng(seed)

filter <- dplyr::filter
arrange <- dplyr::arrange
mutate <- dplyr::mutate
select <- dplyr::select

save_file <- "SP500.RData"
if (file.exists(save_file)) {
  load(save_file)
} else {
  in_sample_begin <- as.Date("1988-01-01")
  out_of_sample_pre_extreme_begin <- as.Date("2017-01-01")
  out_of_sample_extreme_begin <- as.Date("2020-01-01")
  out_of_sample_post_extreme_begin <- as.Date("2020-07-01")
  out_of_sample_end <- as.Date("2021-12-31")
  
  tolerance_grad <- 0.0
  tolerance_change <- 0.0
  ensemble_barrier <- 3.8e-7
  monte_carlo_sample_size <- 20000
  output_diagnostics <- TRUE
  
  timeout_seconds <- 300
  timeout_iterations <- 10000
  
  date_slice <- function(x, begin, end_plus_one) {
    filter(x, Date >= begin, Date < end_plus_one)
  }
  pretty_print_date <- function(t) {
    format(t, format = "%b %-e, %Y")
  }
  
  spxtr <- read_csv("SPXTR.csv", col_types = cols(col_date("%d/%m/%Y"), col_character(), col_double(), col_double()), skip = 2) %>%
    filter(
      Date >= in_sample_begin,
      Date <= out_of_sample_end
    ) %>%
    arrange(Date) %>%
    mutate(`Log S&P 500 Returns` = log(Real_Close) - log(lag(Real_Close))) %>%
    filter(!is.na(`Log S&P 500 Returns`)) %>%
    mutate(t = 0:(n()-1)) %>%
    select(Date, t, `Log S&P 500 Returns`) %>%
    as_tsibble(index = t)
  
  spxtr_in_sample <- spxtr %>% date_slice(in_sample_begin, out_of_sample_pre_extreme_begin)
  spxtr_pre_extreme <- spxtr %>% date_slice(in_sample_begin, out_of_sample_extreme_begin)
  spxtr_extreme <- spxtr %>% date_slice(in_sample_begin, out_of_sample_post_extreme_begin)
  spxtr_post_extreme <- spxtr
  
  in_sample_begin <- min(spxtr_in_sample$Date)
  in_sample_end <- max(spxtr_in_sample$Date)
  out_of_sample_pre_extreme_begin <- min(filter(spxtr, Date >= out_of_sample_pre_extreme_begin)$Date)
  out_of_sample_pre_extreme_end <- max(spxtr_pre_extreme$Date)
  out_of_sample_extreme_begin <- min(filter(spxtr, Date >= out_of_sample_extreme_begin)$Date)
  out_of_sample_extreme_end <- max(spxtr_extreme$Date)
  out_of_sample_post_extreme_begin <- min(filter(spxtr, Date >= out_of_sample_post_extreme_begin)$Date)
  out_of_sample_post_extreme_end <- max(spxtr_post_extreme$Date)
  out_of_sample_overall_begin <- out_of_sample_pre_extreme_begin
  out_of_sample_overall_end <- out_of_sample_post_extreme_end
  
  in_sample_size_pre_extreme <- nrow(spxtr_in_sample)
  in_sample_size_extreme <- nrow(spxtr_pre_extreme)
  in_sample_size_post_extreme <- nrow(spxtr_extreme)
  in_sample_size_overall <- in_sample_size_pre_extreme
  
  out_sample_size_pre_extreme <- in_sample_size_extreme - in_sample_size_pre_extreme
  out_sample_size_extreme <- in_sample_size_post_extreme - in_sample_size_extreme
  out_sample_size_post_extreme <- nrow(spxtr) - in_sample_size_post_extreme
  out_sample_size_overall <- nrow(spxtr) - in_sample_size_overall
  
  overall_sample_size <- nrow(spxtr)
  
  spxtr_plot <- function(x) { ggplot(x, aes(x = Date, y = `Log S&P 500 Returns`)) + geom_line() }
  
  spxtr_libtorch <- libtorch_dict(spxtr %>% select(t, `Log S&P 500 Returns`))
  spxtr_libtorch_in_sample <- libtorch_dict(spxtr_in_sample %>% select(t, `Log S&P 500 Returns`))
  spxtr_libtorch_pre_extreme <- libtorch_dict(spxtr_pre_extreme %>% select(t, `Log S&P 500 Returns`))
  spxtr_libtorch_extreme <- libtorch_dict(spxtr_extreme %>% select(t, `Log S&P 500 Returns`))
  spxtr_libtorch_post_extreme <- spxtr_libtorch
  
  periods <- c(
    paste("Pre-extreme period:", pretty_print_date(out_of_sample_pre_extreme_begin), "to", pretty_print_date(out_of_sample_pre_extreme_end)),
    paste("Extreme period:", pretty_print_date(out_of_sample_extreme_begin), "to", pretty_print_date(out_of_sample_extreme_end)),
    paste("Post-extreme period:", pretty_print_date(out_of_sample_post_extreme_begin), "to", pretty_print_date(out_of_sample_post_extreme_end)),
    paste("Overall period:", pretty_print_date(out_of_sample_pre_extreme_begin), "to", pretty_print_date(out_of_sample_end))
  )
  data_sets <- list(
    spxtr_libtorch_pre_extreme,
    spxtr_libtorch_extreme,
    spxtr_libtorch_post_extreme,
    spxtr_libtorch
  )
  data_sets_train_tibble <- list(
    spxtr_in_sample,
    spxtr_pre_extreme,
    spxtr_extreme,
    spxtr_in_sample
  )
  data_sets_train_libtorch <- list(
    spxtr_libtorch_in_sample,
    spxtr_libtorch_pre_extreme,
    spxtr_libtorch_extreme,
    spxtr_libtorch_in_sample
  )
  min_in_sample_sizes <- c(
    max(spxtr_in_sample$t)+1,
    max(spxtr_pre_extreme$t)+1,
    max(spxtr_extreme$t)+1
  )
  min_in_sample_sizes[4] <- min_in_sample_sizes[1]
  
  periods <- periods[c(4,2)]
  data_sets <- data_sets[c(4,2)]
  data_sets_train_tibble <- data_sets_train_tibble[c(4,2)]
  data_sets_train_libtorch <- data_sets_train_libtorch[c(4,2)]
  min_in_sample_sizes <- min_in_sample_sizes[c(4,2)]
  
  estimators <- c(
    "One-Stage",
    "Two-Stage"
  )
  var_percentages <- c(
    0.05,
    0.01
  )
  
  log_score = LogScore()
  scores <- list()
  for (i in 1:length(periods)) {
    period <- periods[i]
    scores[[period]] <- list(
      LS = log_score,
      CS10 = CensoredLogScore(
        -Inf,
        quantile(data_sets_train_tibble[[i]]$`Log S&P 500 Returns`, 0.1),
        FALSE
      ),
      CS20 = CensoredLogScore(
        -Inf,
        quantile(data_sets_train_tibble[[i]]$`Log S&P 500 Returns`, 0.2),
        FALSE
      )
    )
  }
  score_names <- names(scores[[1]])
  
  untrained_models <- list(
    "AR_1" = ARARCHTX(
      regressand = `Log S&P 500 Returns`,
      ar_intercept_soft_lower_bound = -0.01,
      ar_intercept_soft_upper_bound = 0.01,
      ar_intercept_scaling = 1.0,
      ar_coef_order = 1,
      ar_coef_soft_lower_bound = -0.3,
      ar_coef_soft_upper_bound = 0.3,
      arch_intercept_lower_bound = 0.0,
      arch_intercept_soft_upper_bound = 0.001,
      arch_intercept_scaling = 1.0,
      arch_intercept_barrier_scaling = 1.0,
      arch_coef_order = 0,
      arch_coef_scaling = 1.0,
      var_transformation_crimp = 0.0,
      var_transformation_catch = 0.0,
      tolerance_grad = tolerance_grad,
      tolerance_change = tolerance_change,
      optimiser_timeout_in_seconds = timeout_seconds,
      maximum_optimiser_iterations = timeout_iterations,
      learning_rate = 1e-3,
      barrier_begin = 0.0,
      barrier_end = 0.0
    ),
    "ARCH_1" = ARARCHTX(
      regressand = `Log S&P 500 Returns`,
      ar_intercept = NULL,
      ar_coef_order = 0,
      arch_intercept_lower_bound = 0.0,
      arch_intercept_soft_upper_bound = 0.0004,
      arch_intercept_scaling = 1.0,
      arch_intercept_barrier_scaling = 1.0,
      arch_coef_order = 1,
      arch_coef_lower_bound = 0.0,
      arch_coef_soft_upper_bound = 0.5,
      arch_coef_scaling = 1.0,
      arch_coef_barrier_scaling = 0.0,
      var_transformation_crimp = 0.0,
      var_transformation_catch = 0.0,
      tolerance_grad = tolerance_grad,
      tolerance_change = tolerance_change,
      optimiser_timeout_in_seconds = timeout_seconds,
      maximum_optimiser_iterations = timeout_iterations,
      learning_rate = 1e-3,
      barrier_begin = 0.0,
      barrier_end = 0.0
    )
  )
  model_names <- names(untrained_models)
  
  
  # Train constituents.
  
  trained_models <- list()
  for (i in 1:length(periods)) {
    period <- periods[i]
    data_tibble <- data_sets_train_tibble[[i]]
    data_libtorch <- data_sets_train_libtorch[[i]]
    trained_models[[period]] <- list(LS = list())
    for (model_name in model_names) {
      trained_models[[period]][["LS"]][[model_name]] <- untrained_models[[model_name]]$train(
        data_tibble,
        data_libtorch,
        scores[[period]]$LS,
        return_diagnostics = output_diagnostics
      )
      if (output_diagnostics) {
        diagnostics <- trained_models[[period]][["LS"]][[model_name]]$optimiser_diagnostics[[1]]
        save_plot_list(optimiser_diagnostics_plot_list(diagnostics), file.path("diagnostics", period, model_name, "LS"))
      }
    }
    if (length(score_names) > 1) for (j in 2:length(score_names)) {
      score = scores[[period]][[j]]
      score_name <- score_names[j]
      trained_models[[period]][[score_name]] <- list()
      for (model_name in model_names) {
        trained_models[[period]][[score_name]][[model_name]] <- untrained_models[[model_name]]$retrain(
          trained_models[[period]][["LS"]][[model_name]],
          scoring_rule_retrain = score,
          return_diagnostics = output_diagnostics
        )
        if (output_diagnostics) {
          diagnostics <- trained_models[[period]][[score_name]][[model_name]]$optimiser_diagnostics[[1]]
          save_plot_list(optimiser_diagnostics_plot_list(diagnostics), file.path("diagnostics", period, model_name, score_name))
        }
      }
    }
  }
  
  # Train ensembles.
  
  ararchtx_ensemble <- function(components_in, one_stage) {
    Ensemble(
      components = components_in,
      components_optimise = one_stage,
      components_fixed = FALSE,
      weights_optimise = TRUE,
      weights_fixed = FALSE,
      weights_scaling = 1.0,
      weights_barrier_scaling = 0.0,
      barrier_begin = ensemble_barrier,
      barrier_end = ensemble_barrier,
      tolerance_grad = tolerance_grad,
      tolerance_change = tolerance_change,
      optimiser_timeout_in_seconds = timeout_seconds,
      maximum_optimiser_iterations = timeout_iterations,
      learning_rate = 1e-3
    )
  }
  for (period in periods) {
    for (score_name in score_names) {
      untrained_models[[period]][[score_name]] <- list()
      untrained_models[[period]][[score_name]]$`One-Stage` <- ararchtx_ensemble(trained_models[[period]][[score_name]], TRUE)
      untrained_models[[period]][[score_name]]$`Two-Stage` <- ararchtx_ensemble(trained_models[[period]][[score_name]], FALSE)
    }
  }
  
  for (i in 1:length(periods)) {
    period <- periods[i]
    data_tibble <- data_sets_train_tibble[[i]]
    data_libtorch <- data_sets_train_libtorch[[i]]
    for (j in 1:length(score_names)) {
      score_name <- score_names[j]
      score <- scores[[period]][[j]]
      
      # One-Stage, LS starts at equally weights, all other scores start at LS weights.
      
      if (j == 1) {
        trained_models[[period]][["LS"]]$`One-Stage` <- untrained_models[[period]][["LS"]]$`One-Stage`$train(
          data_tibble,
          data_libtorch,
          scoring_rule_train = score,
          return_diagnostics = output_diagnostics
        )
        if (output_diagnostics) {
          diagnostics <- trained_models[[period]][["LS"]][["One-Stage"]]$optimiser_diagnostics[[1]]
          save_plot_list(optimiser_diagnostics_plot_list(diagnostics), file.path("diagnostics", period, "One-Stage", "LS"))
        }
      } else {
        trained_models[[period]][[score_name]]$`One-Stage` <- untrained_models[[period]][[score_name]]$`One-Stage`$retrain(
          trained_models[[period]][["LS"]]$`One-Stage`,
          scoring_rule_retrain = score,
          return_diagnostics = output_diagnostics
        )
        if (output_diagnostics) {
          diagnostics <- trained_models[[period]][[score_name]][["One-Stage"]]$optimiser_diagnostics[[1]]
          save_plot_list(optimiser_diagnostics_plot_list(diagnostics), file.path("diagnostics", period, "One-Stage", score_name))
        }
      }
      
      # Two-Stage
      
      trained_models[[period]][[score_name]]$`Two-Stage` <- untrained_models[[period]][[score_name]]$`Two-Stage`$train(
        data_tibble,
        data_libtorch,
        scoring_rule_train = score,
        return_diagnostics = output_diagnostics
      )
      if (output_diagnostics) {
        diagnostics <- trained_models[[period]][[score_name]][["Two-Stage"]]$optimiser_diagnostics[[1]]
        save_plot_list(optimiser_diagnostics_plot_list(diagnostics), file.path("diagnostics", period, "Two-Stage", score_name))
      }
    }
  }
  
  sampling_distribution_draws <- list()
  for (period in periods) {
    sampling_distribution_draws[[period]] <- list()
    for (score_name in score_names) {
      sampling_distribution_draws[[period]][[score_name]] <- list()
      for (estimator in estimators) {
        model <- trained_models[[period]][[score_name]][[estimator]]
        sampling_distribution_draws[[period]][[score_name]][[estimator]] <- draw_sampling_distribution(
          sampling_distribution = truncated_kernel_clt(model$model),
          monte_carlo_sample_size
        )
      }
    }
  }
  
  results_table <- list()
  for (i in 1:length(periods)) {
    period <- periods[i]
    data_set <- data_sets[[i]]
    min_in_sample_size <- min_in_sample_sizes[i]
    results_table[[period]] <- list()
    
    # Out of sample scores.
    
    for (j in 1:length(score_names)) {
      test_score_name <- score_names[j]
      test_score <- scores[[period]][[test_score_name]]
      results_table[[period]][[test_score_name]] <- list()
      for (k in 1:length(score_names)) {
        train_score_name <- score_names[k]
        train_score <- scores[[period]][[train_score_name]]
        results_table[[period]][[test_score_name]][[train_score_name]] <- list()
        for (estimator in estimators) {
          res <- list()
          model <- trained_models[[period]][[train_score_name]][[estimator]]
          model_draws <- sampling_distribution_draws[[period]][[train_score_name]][[estimator]]
          res$`Average` <- average_score_out_of_sample(
            model = model,
            observations = data_set,
            scoring_rule = test_score,
            in_sample_times = min_in_sample_size
          )
          avg_score_draws <- average_score_out_of_sample(
            model = model_draws,
            observations = data_set,
            scoring_rule = test_score,
            in_sample_times = min_in_sample_size
          )
          res$`CILow` <- quantile(avg_score_draws, 0.025)
          res$`CIHigh` <- quantile(avg_score_draws, 0.975)
          res$avg_draws <- avg_score_draws
          results_table[[period]][[test_score_name]][[train_score_name]][[estimator]] <- res
        }
      }
    }
    
    # VaR
    
    for (j in 1:length(var_percentages)) {
      perc <- var_percentages[j]
      percstr <- paste0(perc*100, "\\% VaR")
      results_table[[period]][[percstr]] <- list()
      for (k in 1:length(score_names)) {
        train_score_name <- score_names[k]
        train_score <- scores[[period]][[train_score_name]]
        results_table[[period]][[percstr]][[train_score_name]] <- list()
        for (estimator in estimators) {
          res <- list()
          model <- trained_models[[period]][[train_score_name]][[estimator]]
          model_draws <- sampling_distribution_draws[[period]][[train_score_name]][[estimator]]
          res$`Average` <- empirical_coverage(
            models = list(model$model),
            observations_dict = data_set,
            closed_upper_probability = perc,
            in_sample_size = min_in_sample_size
          )
          empirical_coverage_draws <- empirical_coverage(
            models = model_draws,
            observations_dict = data_set,
            closed_upper_probability = perc,
            in_sample_size = min_in_sample_size
          )
          res$`CILow` <- quantile(empirical_coverage_draws, 0.025)
          res$`CIHigh` <- quantile(empirical_coverage_draws, 0.975)
          res$avg_draws <- empirical_coverage_draws
          results_table[[period]][[percstr]][[train_score_name]][[estimator]] <- res
        }
      }
    }
  }
  
  performance_measures <- names(results_table[[1]])
  
  save(
    results_table,
    performance_measures,
    monte_carlo_sample_size,
    periods,
    score_names,
    estimators,
    var_percentages,
    file = save_file
  )
}
