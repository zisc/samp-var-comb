parameters <- function(
  models
) {
  parameters.list.raw <- .Call(
    C_R_parameters,
    models
  )
  
  parameters.list.array <- lapply(
    parameters.list.raw,
    function(parameters.list.raw.this.model) {
      parameters.list.array.this.model <- lapply(
        parameters.list.raw.this.model,
        function(parameter.vector) {
          array.data <- parameter.vector[[1]]
          array.dim <- parameter.vector[[2]]
          return(aperm(array(array.data, array.dim)))
        }
      )
      
      names(parameters.list.array.this.model) <- names(parameters.list.raw.this.model)
    
      return(parameters.list.array.this.model)
    }
  )
  
  return(parameters.list.array)
}

change_parameters <- function(
  model_to_change,
  model_with_parameters
) {
  return(new_libtorch_model_t(
    model = .Call(C_R_change_parameters, model_to_change$model, model_with_parameters$model),
    name = model_to_change$name,
    success = model_to_change$success,
    optimiser_diagnostics = NULL,
    libtorch_data = model_to_change$libtorch_data
  ))
}
