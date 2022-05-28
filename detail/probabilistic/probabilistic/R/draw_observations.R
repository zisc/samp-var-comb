draw_observations <- function(
  model,
  libtorch_dict,
  sample_size,
  burn_in_size = sample_size,
  first_draw = 0.0
) {
  if (!inherits(libtorch_dict, "libtorch_dict")) {
    stop("draw_observations accepts objects of class libtorch_dict only for second argument.")
  }
  return(libtorch_dict_detail(
    dict = .Call(
      C_R_draw_observations,
      model$model,
      as.integer(sample_size),
      as.integer(burn_in_size),
      as.numeric(first_draw)
    ),
    table = libtorch_dict$table,
    tensor = libtorch_dict$tensor
  ))
}
