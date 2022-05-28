train_generic <- function(
  untrained_output,
  model_name,
  scoring_rule,
  learning_rate,
  barrier_begin,
  barrier_end,
  barrier_decay,
  tolerance_grad,
  tolerance_change,
  maximum_optimiser_iterations,
  optimiser_timeout_in_seconds,
  return_diagnostics = TRUE
) {
  models <- NULL
  libtorch_data <- NULL
  if (inherits(untrained_output, "libtorch_model_t")) {
    models <- list(untrained_output$model)
    libtorch_data <- untrained_output$libtorch_data
  } else {
    models <- untrained_output$models
    libtorch_data <- untrained_output$libtorch_data
  }
  
  fitted <- fit(
    models,
    scoring_rule,
    libtorch_data,
    learning_rate,
    barrier_begin,
    barrier_end,
    barrier_decay,
    tolerance_grad,
    tolerance_change,
    maximum_optimiser_iterations,
    optimiser_timeout_in_seconds,
    return_diagnostics
  )
  
  best_model_index <- 1
  if (length(fitted$models) > 1) {
    scores <- sapply(fitted$diagnostics, function(x) { last(x$score) })
    best_model_index_temp <- which.max(scores)
    if (length(best_model_index_temp) > 0) {
      best_model_index <- best_model_index_temp
    }
  }
  
  best_model <- fitted$models[[best_model_index]]
  
  return(new_libtorch_model_t(
    best_model,
    model_name,
    fitted$success[[best_model_index]],
    fitted$diagnostics,
    libtorch_data
  ))
}