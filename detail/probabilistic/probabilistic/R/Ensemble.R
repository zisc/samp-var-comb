Ensemble <- function(
  components,
  components_optimise = TRUE,
  components_fixed = FALSE,
  weights = rep(1, length(components)),
  weights_optimise = TRUE,
  weights_fixed = FALSE,
  weights_name = "weights",
  weights_scaling = 1.0,
  weights_barrier_scaling = 1.0,
  scoring_rule = LogScore(),
  learning_rate = 1.0,
  barrier_begin = 1.0,
  barrier_end = 1e-6,
  barrier_decay = 0.9,
  tolerance_grad = 0.0,
  tolerance_change = 0.0,
  maximum_optimiser_iterations = as.integer(100),
  optimiser_timeout_in_seconds = as.integer(600)
) {
  components_optimise <- as.logical(components_optimise)
  components_fixed <- as.logical(components_fixed)
  weights <- as.numeric(weights)
  weights_optimise <- as.logical(weights_optimise)
  weights_fixed <- as.logical(weights_fixed)
  weights_name <- as.character(weights_name)
  weights_scaling <- as.numeric(weights_scaling)
  weights_barrier_scaling = as.numeric(weights_barrier_scaling)
  
  if (any(weights <= 0)) {
    stop("All weights must be strictly positive.")
  }
  
  if (length(components) == 0) {
    stop("Components is empty.")
  }
  
  if (!weights_optimise && !weights_fixed) {
    stop("If weights is not to be optimised, then it must be treated as fixed.")
  }
  
  if (length(components) == 1) {
    return(components[[1]])
  }
  
  untrained <- function(tsibble_data, libtorch_data) {
    parameter_guesses = list(list(
      weights = shapely.parameter(
        parameter = weights,
        parameter_scaling = weights_scaling,
        barrier_scaling = weights_barrier_scaling,
        enable = TRUE
      )
    ))
    names(parameter_guesses[[1]]) <- weights_name
    
    buffers <- list(weights_optimise, weights_fixed, components_optimise, components_fixed)
    
    models <- list(
      models = .Call(C_R_ManufactureEnsemble,
        lapply(components, function(c) { c$model }),
        parameter_guesses,
        buffers
      ),
      libtorch_data = libtorch_data
    )
    
    return(models)
  }
  
  train <- function(
    tsibble_data,
    libtorch_data = libtorch_dict(),
    scoring_rule_train = scoring_rule,
    learning_rate_train = learning_rate,
    barrier_begin_train = barrier_begin,
    barrier_end_train = barrier_end,
    barrier_decay_train = barrier_decay,
    tolerance_grad_train = as.numeric(tolerance_grad),
    tolerance_change_train = as.numeric(tolerance_change),
    maximum_optimiser_iterations_train = maximum_optimiser_iterations,
    optimiser_timeout_in_seconds_train = optimiser_timeout_in_seconds,
    return_diagnostics = TRUE
  ) {
    ensemble_name <- "Ensemble("
    for (c in components) {
      ensemble_name <- paste0(ensemble_name, c$name, ", ")
    }
    ensemble_name <- paste0(
      ensemble_name,
      "components_optimise = ", components_optimise, ", ",
      "components_fixed = ", components_fixed, ", ",
      "weights = c(", weights[1]
    )
    for (w in weights[2:length(weights)]) {
      ensemble_name <- paste0(ensemble_name, ", ", w)
    }
    ensemble_name <- paste0(
      ensemble_name,
      "), ",
      "weights_optimise = ", weights_optimise, ", ",
      "weights_fixed = ", weights_fixed,
      ")"
    )
    tg <- train_generic(
      untrained(tsibble_data, libtorch_data),
      ensemble_name,
      scoring_rule_train,
      learning_rate_train,
      barrier_begin_train,
      barrier_end_train,
      barrier_decay_train,
      as.numeric(tolerance_grad_train),
      as.numeric(tolerance_change_train),
      maximum_optimiser_iterations_train,
      optimiser_timeout_in_seconds_train,
      return_diagnostics
    )
    libtorch_data <<- tg$libtorch_data
    return(tg)
  }
  
  retrain <- function(
    libtorch_model,
    libtorch_data = libtorch_model$libtorch_data,
    scoring_rule_retrain = scoring_rule,
    learning_rate_retrain = learning_rate,
    barrier_begin_retrain = barrier_begin,
    barrier_end_retrain = barrier_end,
    barrier_decay_retrain = barrier_decay,
    tolerance_grad_retrain = tolerance_grad,
    tolerance_change_retrain = tolerance_change,
    maximum_optimiser_iterations_retrain = maximum_optimiser_iterations,
    optimiser_timeout_in_seconds_retrain = optimiser_timeout_in_seconds,
    return_diagnostics = TRUE
  ) {
    tg <- train_generic(
      list(models = list(libtorch_model$model), libtorch_data = libtorch_data),
      libtorch_model$name,
      scoring_rule_retrain,
      learning_rate_retrain,
      barrier_begin_retrain,
      barrier_end_retrain,
      barrier_decay_retrain,
      as.numeric(tolerance_grad_retrain),
      as.numeric(tolerance_change_retrain),
      maximum_optimiser_iterations_retrain,
      optimiser_timeout_in_seconds_retrain,
      return_diagnostics
    )
    libtorch_data <<- tg$libtorch_data
    return(tg)
  }
  
  return(list(
    untrained = untrained,
    train = train,
    retrain = retrain
  ))
}

change_components <- function(ensemble, new_components) {
  if (!inherits(ensemble, "libtorch_model_t")) {
    stop("change_components only operators on trained ensembles")
  }
  
  return(new_libtorch_model_t(
    model = .Call(
      C_R_change_components,
      ensemble$model,
      lapply(new_components, function(c) { c$model })
    ),
    name = ensemble$name,
    success = ensemble$success,
    optimiser_diagnostics = NULL,
    libtorch_data = ensemble$libtorch_data
  ))
}
