draw_sampling_distribution <- function(
  sampling_distribution,
  num_draws
) {
  return(.Call(C_R_sampling_distribution_draws,
    sampling_distribution,
    as.integer(num_draws)
  ))
}