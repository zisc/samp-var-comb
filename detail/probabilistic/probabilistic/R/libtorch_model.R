new_libtorch_model_t <- function(model, name, success, optimiser_diagnostics, libtorch_data) {
  structure(
    list(
      model = model,
      name = name,
      success = success,
      optimiser_diagnostics = optimiser_diagnostics,
      libtorch_data = libtorch_data
    ),
    class = c("libtorch_model_t", "list")
  )
}

new_libtorch_model_collection_t <- function(
  libtorch_data,
  models
) {
  structure(
    list(
      libtorch_data = libtorch_data,
      models = models
    ),
    class = c("libtorch_model_collection_t", "list")
  )
}

libtorch_model <- function(data, ...) {
  libtorch_data <- libtorch_dict()
  models <- lapply(
    list(...),
    function(closures) { closures$train(libtorch_data, data) }
  )
  return(new_libtorch_model_collection_t(libtorch_data, models))
}

serialise_libtorch_model <- function(libtorch_model) {
  structure(
    list(
      model = .Call(C_R_serialise_model, list(libtorch_model$model))[[1]],
      name = libtorch_model$name,
      optimiser_diagnostics = libtorch_model$optimiser_diagnostics
    ),
    class = c("libtorch_model_serialised_t", "list")
  )
}

deserialise_libtorch_model <- function(libtorch_model_serialised, model_out) {
  .Call(C_R_deserialise_model,
    model_out$models,
    list(libtorch_model_serialised$model)
  )
  return(new_libtorch_model_t(
    model = model_out$models[[1]],
    name = libtorch_model_serialised$name,
    optimiser_diagnostics = libtorch_model_serialised$optimiser_diagnostics,
    libtorch_data = model_out$libtorch_data
  ))
}
