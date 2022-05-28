get_model_coerced <- function(model) {
  model_coerced <- NULL
  if (inherits(model, "libtorch_model_t")) {
    model_coerced <- list(model$model)
  } else if (is_xptr(model)) {
    model_coerced <- list(model)
  } else if (is.list(model)) {
    model_coerced <- model
  } else {
    stop("model type unrecognised, pass a libtorch_model_t, an extptrsxp, or a list of extptrsxp elements.")
  }
  return(model_coerced)
}

average_score <- function(
  model,
  observations = NULL,
  scoring_rule = NULL
) {
  return(.Call(
    C_R_average_score,
    get_model_coerced(model),
    observations$dict,
    scoring_rule
  ))
}

average_score_out_of_sample <- function(
  model,
  observations = NULL,
  scoring_rule = NULL,
  in_sample_times = 0
) {
  return(.Call(
    C_R_average_score_out_of_sample,
    get_model_coerced(model),
    observations$dict,
    scoring_rule,
    as.integer(in_sample_times)
  ))
}
