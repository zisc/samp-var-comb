serialise.model <- function(models) {
  return(.Call(
    C_R_serialise_model,
    models
  ))
}

deserialise.model <- function(models.out, serialised.models) {
  return(.Call(
    C_R_deserialise_model,
    models.out,
    serialised.models
  ))
}
