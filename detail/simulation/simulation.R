library(DBI)
library(grid)
library(gridExtra)
library(lemon)
library(latex2exp)
library(MASS)
library(Rfast)
library(pracma)
library(tsibble)
library(probabilistic)

source("momentify.R")

seed_file <- "seed.txt"
if (!file.exists(seed_file)) {
  write(as.integer(Sys.time()), file = seed_file)
}
seed <- scan(file = seed_file, what = integer(), quiet = TRUE)
set.seed(seed)
seed_torch_rng(seed)

dgp_trim <- 5
trim_dgp_draws <- function(libtorch_draws) {
  tsibble_dict <- libtorch_dict_to_tables(libtorch_draws)[[1]] %>%
    mutate(y = pmin(pmax(y, -dgp_trim), dgp_trim))
  return(libtorch_dict(tsibble_dict))
}

ensemble_score_experiment <- function(
  dgp,
  constituent_models,  # list
  scores,              # list
  me_sample_sizes = as.integer(round(seq(from = 500, to = 2000, length.out = 40))),
  opt_sample_size = as.integer(250000),
  dgp_scores_sample_size = as.integer(1000000),
  score_sample_size_coef = 100,
  num_mes = 1000,
  libtorch_dict = dgp$libtorch_data,
  simulation_confidence_level = 0.95,
  ensemble_weights_scaling = 1.0,
  ensemble_weights_barrier_scaling = 1.0,
  ensemble_learning_rate = 0.01,
  ensemble_learning_rate_retry = 1e-4,
  ensemble_barrier_begin = 0.01,
  ensemble_barrier_begin_retry = 1.0,
  ensemble_barrier_end_coef = 200,
  ensemble_barrier_end_exp = -3,
  ensemble_barrier_end_max = 0.01,
  ensemble_barrier_decay = 0.1,
  ensemble_barrier_decay_retry = 0.9,
  ensemble_tolerance_grad = 2e-6,
  ensemble_tolerance_change = 2e-7,
  ensemble_optimiser_timeout_in_seconds = 60,
  ensemble_maximum_optimiser_iterations = 100000,
  ensemble_optimiser_timeout_in_seconds_retry = 600,
  ensemble_maximum_optimiser_iterations_retry = 100000,
  ensemble_tuning_special_cases = list(),
  db_file = "simulation.sqlite"
) {
  me_sample_sizes <- as.integer(me_sample_sizes)
  names(scores) <- sapply(scores, function(s) { attr(s, "name") })
  
  first_run <- !file.exists(db_file)
  db <- dbConnect(RSQLite::SQLite(), db_file)
  if (first_run) {
    dbBegin(db)
    dbWriteTable(db, "next_me", tibble(next_me = as.integer(1)))
    dbWriteTable(db, "R_rng_state", tibble(R_rng_state = .Random.seed))
    dbWriteTable(db, "torch_rng_state", tibble(torch_rng_state = list(get_state_torch_rng())))
    dbCommit(db)
  }
  next_me <- dbReadTable(db, "next_me")$next_me
  
  process_results <- function() {
    .Random.seed <- dbReadTable(db, "R_rng_state")$R_rng_state
    set_state_torch_rng(dbReadTable(db, "torch_rng_state")$torch_rng_state[[1]])
    
    num_scores <- length(scores)
    dgp_scores <- tibble(
      Measure = rep(NA_character_, num_scores),
      DGPScore = rep(NA_real_, num_scores)
    )
    dgp_scores_sample_dict <- trim_dgp_draws(draw_observations(dgp, libtorch_dict, dgp_scores_sample_size))
    names_scores <- names(scores)
    i <- as.integer(1)
    for (score_measured_nm in names_scores) {
      dgp_scores$Measure[i] <- score_measured_nm
      dgp_scores$DGPScore[i] <- average_score(dgp, dgp_scores_sample_dict, scores[[score_measured_nm]])
      i <- i + as.integer(1)
    }
    
    results_detail <- dbReadTable(db, "results_detail") %>%
      rename(`Sample Size` = `Sample.Size`)
    
    results_sans_moments <- list(
      detail = results_detail,
      dgp_scores = dgp_scores,
      simulation_confidence_level = simulation_confidence_level
    )
    
    return(momentify(results_sans_moments))
  }
  
  if (next_me > num_mes) {
    results <- process_results()
    dbDisconnect(db)
    return(results)
  }
  
  get_ensemble_barrier_end <- function(n) {
    min(ensemble_barrier_end_max, ensemble_barrier_end_coef*(n^ensemble_barrier_end_exp))
  }
  
  score_sample_size <- function(n) {
    score_sample_size_coef*n
  }
  
  num_constituent_models = length(constituent_models)
  num_scores = length(scores)
  
  me_sample_sizes <- sort(me_sample_sizes, decreasing = TRUE)
  total_sample_size <- 2*(max(me_sample_sizes) + score_sample_size(max(me_sample_sizes)))
  num_me_sample_sizes <- length(me_sample_sizes)
  
  opt_sample_dict <- trim_dgp_draws(draw_observations(dgp, libtorch_dict, opt_sample_size))
  opt_sample_tsibble <- libtorch_dict_to_tables(opt_sample_dict)[[1]]
  
  opt_models <- lapply(
    scores,
    function(s) {
      lapply(
        constituent_models,
        function(m) {
          ret <- m$train(
            tsibble_data = opt_sample_tsibble,
            libtorch_data = opt_sample_dict,
            scoring_rule_train = s,
            barrier_begin_train = ensemble_barrier_begin,
            barrier_end_train = get_ensemble_barrier_end(opt_sample_size),
            barrier_decay = ensemble_barrier_decay,
            tolerance_grad = 0.0,
            tolerance_change = 0.0,
            return_diagnostics = FALSE
          )
          if (!ret$success) {
            stop("Optimisation unsuccessful.")
          }
          return(ret)
        }
      )
    }
  )
  
  EnsembleTemplate <- function(components, components_optimise, weights_optimise) {
    Ensemble(
      components = components,
      components_optimise = components_optimise,
      weights_optimise = weights_optimise,
      weights_scaling = ensemble_weights_scaling,
      weights_barrier_scaling = ensemble_weights_barrier_scaling,
      barrier_decay = ensemble_barrier_decay,
      learning_rate = ensemble_learning_rate,
      tolerance_grad = ensemble_tolerance_grad,
      tolerance_change = ensemble_tolerance_change,
      optimiser_timeout_in_seconds = 10*ensemble_optimiser_timeout_in_seconds,
      maximum_optimiser_iterations = 10*ensemble_maximum_optimiser_iterations
    )
  }
  
  ensembles <- lapply(
    scores,
    function(s) {
      om <- opt_models[[attr(s, "name")]]
      return(list(
        `One-Stage` = EnsembleTemplate(
          components = om,
          components_optimise = TRUE,
          weights_optimise = TRUE
        ),
        `Two-Stage` = EnsembleTemplate(
          components = om,
          components_optimise = FALSE,
          weights_optimise = TRUE
        )
      ))
    }
  )
  
  opt_ensembles <- lapply(
    scores,
    function(s) {
      lapply(
        ensembles[[attr(s, "name")]],
        function(e) {
          ret <- e$train(
            opt_sample_tsibble,
            opt_sample_dict,
            s,
            barrier_begin_train = ensemble_barrier_begin,
            barrier_end_train = get_ensemble_barrier_end(opt_sample_size),
            tolerance_grad_train = 0.0,
            tolerance_change_train = 0.0,
            return_diagnostics = FALSE
          )
          if (!ret$success) {
            stop("Optimisation unsucessful.")
          }
          return(ret)
        }
      )
    }
  )
  
  opt_ensemble_params <- lapply(
    scores,
    function(s) {
      lapply(
        opt_ensembles[[attr(s, "name")]],
        function(oe) {
          unlist(parameters(list(oe$model)))
        }
      )
    }
  )
  
  get_learning_rate <- function(
    me_num,
    sample_size,
    score_optimised,
    estimator
  ) {
    learning_rate <- ensemble_tuning_special_cases[[as.character(me_num)]][[as.character(sample_size)]][[score_optimised]][[estimator]][["learning_rate"]]
    if (is.null(learning_rate)) {
      return(ensemble_learning_rate)
    } else {
      print(paste0("Special case: learning rate = ", learning_rate, " for me = ", me_num, ", n = ", sample_size, ", score = ", score_optimised, ", estimator = \"", estimator, "\"."))
      return(learning_rate)
    }
  }
  
  get_barrier_begin <- function(
    me_num,
    sample_size,
    score_optimised,
    estimator
  ) {
    barrier_begin <- ensemble_tuning_special_cases[[as.character(me_num)]][[as.character(sample_size)]][[score_optimised]][[estimator]][["barrier_begin"]]
    if (is.null(barrier_begin)) {
      return(get_ensemble_barrier_end(n))
    } else {
      print(paste0("Special case: barrier_begin = ", barrier_begin, " for me = ", me_num, ", n = ", sample_size, ", score = ", score_optimised, ", estimator = \"", estimator, "\"."))
      return(barrier_begin)
    }
  }
  
  get_barrier_decay <- function(
    me_num,
    sample_size,
    score_optimised,
    estimator
  ) {
    barrier_decay <- ensemble_tuning_special_cases[[as.character(me_num)]][[as.character(sample_size)]][[score_optimised]][[estimator]][["barrier_decay"]]
    if (is.null(barrier_decay)) {
      return(ensemble_barrier_decay)
    } else {
      print(paste0("Special case: barrier_decay = ", barrier_decay, " for me = ", me_num, ", n = ", sample_size, ", score = ", score_optimised, ", estimator = \"", estimator, "\"."))
    }
  }
  
  .Random.seed <- dbReadTable(db, "R_rng_state")$R_rng_state
  set_state_torch_rng(dbReadTable(db, "torch_rng_state")$torch_rng_state[[1]])
  names_scores <- names(scores)
  if (next_me <= num_mes) for (me in next_me:num_mes) {
    print(paste0("me = ", me))
    start_time <- Sys.time()
    
    total_sample_dict <- trim_dgp_draws(draw_observations(dgp, libtorch_dict, total_sample_size))
    
    i <- as.integer(1)
    num_results_detail <- 3*num_scores*num_scores*num_me_sample_sizes
    results_detail <- tibble(
      Estimator = rep(NA_character_, num_results_detail),
      Optimise = rep(NA_character_, num_results_detail),
      Measure = rep(NA_character_, num_results_detail),
      `Sample Size` = rep(NA_integer_, num_results_detail),
      Score = rep(NA_real_, num_results_detail)
    )
    
    # Using the optimial model as our initial guess, we first estimate using the
    # highest sample size. Using this model as our initial guess, we then estimate
    # using the second highest sample size. And so on. This will hopefully ensure
    # that each sucessive model is only a small distance away from the last,
    # speeding up estimation. It should also mean that we can leave the barrier
    # where it is, speeding up estimation further.
    
    # These next variables will remember the previously estimated model, which
    # forms the initial guess of the current model.
    
    one_stage <- lapply(
      opt_ensembles,
      function(oe) { oe$`One-Stage` }
    )
    
    two_stage_constituents <- opt_models
    two_stage <- lapply(
      opt_ensembles,
      function(oe) { oe$`Two-Stage`}
    )
    
    for (n in me_sample_sizes) {
      print(paste0("n = ", n))
      train_sample_dict <- time_slice(total_sample_dict, 0, n)
      score_sample_dict <- time_slice(total_sample_dict, total_sample_size - score_sample_size(n), total_sample_size)
      ensemble_barrier_end_n <- get_ensemble_barrier_end(n)
      for (score_optimised_nm in names_scores) {
        score_optimised <- scores[[score_optimised_nm]]
        
        one_stage_retrain <- function(
          to_retrain,
          learning_rate,
          barrier_begin,
          barrier_decay,
          optimiser_timeout_in_seconds,
          maximum_optimiser_iterations,
          return_diagnostics = FALSE
        ) {
          ensembles[[score_optimised_nm]]$`One-Stage`$retrain(
            to_retrain,
            # one_stage[[score_optimised_nm]],
            train_sample_dict,
            score_optimised,
            learning_rate_retrain = learning_rate,
            barrier_begin_retrain = barrier_begin,
            barrier_end_retrain = ensemble_barrier_end_n,
            barrier_decay_retrain = barrier_decay,
            tolerance_change_retrain = ensemble_tolerance_change*(learning_rate/ensemble_learning_rate),
            optimiser_timeout_in_seconds = optimiser_timeout_in_seconds,
            maximum_optimiser_iterations = maximum_optimiser_iterations,
            return_diagnostics = return_diagnostics
          )
        }
        one_stage_nested <- one_stage_retrain(
          one_stage[[score_optimised_nm]],
          get_learning_rate(me, n, score_optimised_nm, "One-Stage"),
          get_barrier_begin(me, n, score_optimised_nm, "One-Stage"),
          get_barrier_decay(me, n, score_optimised_nm, "One-Stage"),
          ensemble_optimiser_timeout_in_seconds,
          ensemble_maximum_optimiser_iterations
        )
        if (!one_stage_nested$success) {
          print(paste0("One-stage retrain failed at me = ", me, ", n = ", n, ", score = ", score_optimised_nm, ". Retrying with a conservative barrier sequence."))
          one_stage_nested <- one_stage_retrain(
            one_stage[[score_optimised_nm]],
            ensemble_learning_rate_retry,
            ensemble_barrier_begin_retry,
            ensemble_barrier_decay_retry,
            ensemble_optimiser_timeout_in_seconds_retry,
            ensemble_maximum_optimiser_iterations_retry,
            TRUE
          )
          if (!one_stage_nested$success) {
            save_plot_list(optimiser_diagnostics_plot_list(one_stage_nested$optimiser_diagnostics[[1]]), paste0("diagnostics_onestage_me", me, "_n", n, "_score", gsub(" ", "", score_optimised_nm, fixed = TRUE)))
            stop("Retry failed, saved diagnostics.")
          }
        }
        one_stage[[score_optimised_nm]] <- one_stage_nested
        
        for (model_num in 1:length(constituent_models)) {
          constituent_retrain <- function(
            to_retrain,
            learning_rate,
            barrier_begin,
            barrier_decay,
            optimiser_timeout_in_seconds,
            maximum_optimiser_iterations,
            return_diagnostics = FALSE
          ) {
            constituent_models[[model_num]]$retrain(
              to_retrain,
              train_sample_dict,
              score_optimised,
              learning_rate_retrain = learning_rate,
              barrier_begin_retrain = barrier_begin,
              barrier_end_retrain = ensemble_barrier_end_n,
              barrier_decay_retrain = barrier_decay,
              tolerance_grad_retrain = ensemble_tolerance_grad,
              tolerance_change_retrain = ensemble_tolerance_change*(learning_rate/ensemble_learning_rate),
              optimiser_timeout_in_seconds = optimiser_timeout_in_seconds,
              maximum_optimiser_iterations = maximum_optimiser_iterations,
              return_diagnostics = return_diagnostics
            )
          }
          const_str <- paste0("Constituent", model_num)
          constituent_nested <- constituent_retrain(
            two_stage_constituents[[score_optimised_nm]][[model_num]],
            get_learning_rate(me, n, score_optimised_nm, const_str),
            get_barrier_begin(me, n, score_optimised_nm, const_str),
            get_barrier_decay(me, n, score_optimised_nm, const_str),
            ensemble_optimiser_timeout_in_seconds,
            ensemble_maximum_optimiser_iterations
          )
          if (!constituent_nested$success) {
            print(paste0("Constituent model ", model_num, " retrain failed at me = ", me, ", n = ", n, ", score = ", score_optimised_nm, ". Retrying with a conservative barrier sequence."))
            constituent_nested <- constituent_retrain(
              two_stage_constituents[[score_optimised_nm]][[model_num]],
              ensemble_learning_rate_retry,
              ensemble_barrier_begin_retry,
              ensemble_barrier_decay_retry,
              ensemble_optimiser_timeout_in_seconds_retry,
              ensemble_maximum_optimiser_iterations_retry,
              TRUE
            )
            if (!one_stage_nested$success) {
              save_plot_list(optimiser_diagnostics_plot_list(constituent_nested$optimiser_diagnostics[[1]], paste0("diagnostics_const", model_num, "_me", me, "_n", n, "_score", gsub(" ", "", score_optimised_nm, fixed = TRUE))))
              stop("Retry failed, saved diagnostics.")
            }
          }
          two_stage_constituents[[score_optimised_nm]][[model_num]] <- constituent_nested
        }
        
        two_stage_optimal_weights <- change_components(
          opt_ensembles[[score_optimised_nm]]$`Two-Stage`,
          two_stage_constituents[[score_optimised_nm]]
        )
        
        two_stage_fit_constituents <- change_components(
          two_stage[[score_optimised_nm]],
          two_stage_constituents[[score_optimised_nm]]
        )
        
        two_stage_retrain <- function(
          to_retrain,
          learning_rate,
          barrier_begin,
          barrier_decay,
          optimiser_timeout_in_seconds,
          maximum_optimiser_iterations,
          return_diagnostics = FALSE
        ) {
          ensembles[[score_optimised_nm]]$`Two-Stage`$retrain(
            to_retrain,
            train_sample_dict,
            score_optimised,
            learning_rate_retrain = learning_rate,
            barrier_begin_retrain = barrier_begin,
            barrier_end_retrain = ensemble_barrier_end_n,
            barrier_decay_retrain = barrier_decay,
            tolerance_grad_retrain = ensemble_tolerance_grad,
            tolerance_change_retrain = ensemble_tolerance_change*(learning_rate/ensemble_learning_rate),
            optimiser_timeout_in_seconds = optimiser_timeout_in_seconds,
            maximum_optimiser_iterations = maximum_optimiser_iterations,
            return_diagnostics = return_diagnostics
          )
        }
        two_stage_nested <- two_stage_retrain(
          two_stage_fit_constituents,
          get_learning_rate(me, n, score_optimised_nm, "Two-Stage"),
          get_barrier_begin(me, n, score_optimised_nm, "Two-Stage"),
          get_barrier_decay(me, n, score_optimised_nm, "Two-Stage"),
          ensemble_optimiser_timeout_in_seconds,
          ensemble_maximum_optimiser_iterations
        )
        if (!two_stage_nested$success) {
          print(paste0("Two-stage retrain failed at me = ", me, ", n = ", n, ", score = ", score_optimised_nm, ". Retrying with a conservative barrier sequence."))
          two_stage_nested <- two_stage_retrain(
            # two_stage_optimal_weights,
            two_stage_fit_constituents,
            ensemble_learning_rate_retry,
            ensemble_barrier_begin_retry,
            ensemble_barrier_decay_retry,
            ensemble_optimiser_timeout_in_seconds_retry,
            ensemble_maximum_optimiser_iterations_retry,
            TRUE
          )
          if (!two_stage_nested$success) {
            save_plot_list(optimiser_diagnostics_plot_list(two_stage_nested$optimiser_diagnostics[[1]], paste0("diagnostics_twostage_me", me, "_n", n, "_score", gsub(" ", "", score_optimised_nm, fixed = TRUE))))
            stop("Retry failed, saved diagnostics.")
          }
        }
        two_stage[[score_optimised_nm]] <- two_stage_nested
        
        for (score_measured_nm in names_scores) {
          score_measured <- scores[[score_measured_nm]]
          
          results_detail$Estimator[i] <- "One-Stage"
          results_detail$Optimise[i] <- score_optimised_nm
          results_detail$Measure[i] <- score_measured_nm
          results_detail$`Sample Size`[i] <- n
          results_detail$Score[i] <- average_score(one_stage[[score_optimised_nm]], score_sample_dict, score_measured)
          i <- i + as.integer(1)
          
          results_detail$Estimator[i] <- "Two-Stage - Weights Fixed at Limit Optimiser"
          results_detail$Optimise[i] <- score_optimised_nm
          results_detail$Measure[i] <- score_measured_nm
          results_detail$`Sample Size`[i] <- n
          results_detail$Score[i] <- average_score(two_stage_optimal_weights, score_sample_dict, score_measured)
          i <- i + as.integer(1)
          
          results_detail$Estimator[i] <- "Two-Stage"
          results_detail$Optimise[i] <- score_optimised_nm
          results_detail$Measure[i] <- score_measured_nm
          results_detail$`Sample Size`[i] <- n
          results_detail$Score[i] <- average_score(two_stage[[score_optimised_nm]], score_sample_dict, score_measured)
          i <- i + as.integer(1)
        }
      }
    }
    if ((me %% 100) == 0) {
      gc()
    }
    
    dbBegin(db)
    dbWriteTable(db, "results_detail", results_detail, append = TRUE)
    dbWriteTable(db, "next_me", tibble(next_me = as.integer(me+1)), overwrite = TRUE)
    dbWriteTable(db, "R_rng_state", tibble(R_rng_state = .Random.seed), overwrite = TRUE)
    dbWriteTable(db, "torch_rng_state", tibble(torch_rng_state = list(get_state_torch_rng())), overwrite = TRUE)
    dbCommit(db)
    
    end_time <- Sys.time()
    
    print(end_time - start_time)
  }
  
  results <- process_results()
  dbDisconnect(db)
  return(results)
}

save_file <- "simulation.RData"
if (file.exists(save_file)) {
  load(save_file)
} else {
  obs_str <- tsibble(
    t = as.integer(c(0,1)),
    y = rep(as.numeric(0.0), 2),
    index = t
  )
  
  dgp <- ARARCHTX(
    regressand = y,
    ar_intercept_init = 0.0,
    ar_intercept_scaling = 1.0,
    ar_coef_init = 0.5,
    ar_coef_scaling = 1.0,
    arch_intercept_init = 0.2,
    arch_intercept_scaling = 1.0,
    arch_coef_init = 0.75,
    arch_coef_scaling = 1.0,
    pre_optimise = FALSE,
    var_transformation_crimp = 0.0,
    var_transformation_catch = 1.0,
    barrier_begin = 0.0,
    barrier_end = 0.0
  )$untrained(obs_str)
  
  dgp_sample <- libtorch_dict_to_tables(trim_dgp_draws(draw_observations(dgp, dgp$libtorch_data, 10000000)))[[1]]
  prop_outlier <- (sum(dgp_sample$y <= -dgp_trim) + sum(dgp_sample$y >= dgp_trim))/10000000
  sd_dgp <- sd(dgp_sample$y)
  scores <- list(
    LogScore(),
    CensoredLogScore(-Inf, quantile(dgp_sample$y, 0.2), complement = FALSE)
  )
  
  # Inspect stationary distribution of DGP
  # dgp_ecdf <- ecdf(dgp_sample$y)
  # PB <- dgp_ecdf(-1) + 1 - dgp_ecdf(1)
  # not_stationary_density <- tibble(x = seq(-10, 10, 0.01)) %>%
  #   mutate(y = dnorm(x, 0, sqrt(var(dgp_sample$y))))
  # dgp_histogram <- ggplot(dgp_sample, aes(y)) +
  #   geom_histogram(aes(y = ..density..), bins = 1000) +
  #   geom_line(aes(x=x, y=y), not_stationary_density)
  # Notice in the above plot that the stationary density is not normal with the mean and variance matched
  # to the stationary mean and variance.
  
  dgp_sample <- NULL
  gc()
  
  constituent_models <- list(
    ARARCHTX(
      regressand = y,
      pre_optimise = FALSE,
      ar_intercept_init = 0.0,
      ar_intercept_soft_lower_bound = -100.0,
      ar_intercept_soft_upper_bound = 100.0,
      ar_intercept_scaling = 1.0,
      ar_intercept_barrier_scaling = 1.0,
      ar_coef_init = 0.6,
      ar_coef_soft_lower_bound = -100.0,
      ar_coef_soft_upper_bound = 100.0,
      ar_coef_scaling = 1.0,
      ar_coef_barrier_scaling = 1.0,
      arch_intercept_init = 1.0,
      arch_intercept_soft_upper_bound = 100.0,
      arch_intercept_scaling = 1.0,
      arch_intercept_barrier_scaling = 1.0,
      arch_coef_order = 0,
      var_transformation_crimp = 0.0,
      var_transformation_catch = 1.0,
      learning_rate = 0.01,
      barrier_begin = 0.0,
      barrier_end = 0.0
    ),
    ARARCHTX(
      regressand = y,
      pre_optimise = FALSE,
      ar_intercept_init = 0.0,
      ar_intercept_soft_lower_bound = 100.0,
      ar_intercept_soft_upper_bound = -100.0,
      ar_intercept_scaling = 1.0,
      ar_intercept_barrier_scaling = 1.0,
      ar_coef_order = 0,
      arch_intercept_init = 1.0,
      arch_intercept_soft_upper_bound = 100.0,
      arch_intercept_scaling = 1.0,
      arch_intercept_barrier_scaling = 1.0,
      arch_coef_init = 0.4,
      arch_coef_soft_upper_bound = 100.0,
      arch_coef_scaling = 1.0,
      arch_coef_barrier_scaling = 1.0,
      var_transformation_crimp = 0.0,
      var_transformation_catch = 1.0,
      learning_rate = 0.01,
      barrier_begin = 0.0,
      barrier_end = 0.0,
      barrier_decay = 0.0
    )
  )
  
  simulation_results <<- ensemble_score_experiment(
    dgp,
    constituent_models,
    scores,
    ensemble_tuning_special_cases = list()
  )
  
  save(simulation_results, file = save_file)
}
