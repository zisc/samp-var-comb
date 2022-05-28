empirical_coverage <- function(
  models,
  observations_dict,
  open_lower_probability = -Inf,
  closed_upper_probability = Inf,
  complement = FALSE,
  in_sample_size
) {
  return(.Call(C_R_empirical_coverage,
    models,
    observations_dict$dict,
    as.numeric(open_lower_probability),
    as.numeric(closed_upper_probability),
    as.logical(complement),
    as.integer(in_sample_size)
  ))
}

empirical_coverage_expanding_window <- function(
  model,
  observations_dict = NULL,
  open_lower_probability = -Inf,
  closed_upper_probability = Inf,
  complement = FALSE,
  min_in_sample_size
) {
  if (is.null(observations_dict)) {
    return(.Call(C_R_empirical_coverage_expanding_window_noobs,
      model,
      as.numeric(open_lower_probability),
      as.numeric(closed_upper_probability),
      as.logical(complement),
      as.integer(min_in_sample_size)
    ))
  } else {
    return(.Call(C_R_empirical_coverage_expanding_window_obs,
      model,
      observations_dict$dict,
      as.numeric(open_lower_probability),
      as.numeric(closed_upper_probability),
      as.logical(complement),
      as.integer(min_in_sample_size)
    ))
  }
}
