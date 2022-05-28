fit <- function(
  models,
  scoring_rule,
  observations_dict,
  learning_rate =  0.02,
  barrier_begin = 1.0,
  barrier_end = 1e-6,
  barrier_decay = 0.99,
  tolerance_grad = 0.0,
  tolerance_change = 0.0,
  maximum_optimiser_iterations = as.integer(10000),
  timeout_in_seconds = as.integer(600),
  return_diagnostics = TRUE
) {
  return(.Call(C_R_fit,
    models,
    scoring_rule,
    observations_dict$dict,
    as.numeric(learning_rate),
    as.numeric(barrier_begin),
    as.numeric(barrier_end),
    as.numeric(barrier_decay),
    as.numeric(tolerance_grad),
    as.numeric(tolerance_change),
    as.integer(maximum_optimiser_iterations),
    as.integer(timeout_in_seconds),
    as.logical(return_diagnostics)
  ))
}
