draw_performance_divergence <- function(
  sampling_distribution,
  scoring_rule = NULL,
  num_draws
) {
  return(.Call(C_R_performance_divergence_draws,
    sampling_distribution,
    scoring_rule,
    as.integer(num_draws)
  ))
}