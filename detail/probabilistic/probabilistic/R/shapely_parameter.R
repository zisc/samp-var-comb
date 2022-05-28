shapely.parameter <- function(
  parameter = NaN,
  lower_bound = NaN,
  upper_bound = NaN,
  parameter_scaling = NaN,
  barrier_scaling = NaN,
  enable = TRUE
) {
  return(list(
    as.numeric(parameter),
    as.numeric(lower_bound),
    as.numeric(upper_bound),
    as.numeric(parameter_scaling),
    as.numeric(barrier_scaling),
    as.logical(enable)
  ))
}
