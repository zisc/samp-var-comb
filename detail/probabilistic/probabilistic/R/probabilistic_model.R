# These two functions are currently not in use.
# it has proven easier to just hold a list of
# external pointers, but this is not type safe.
# Unfortunately, indexing a list with [] that
# has a class does not preserve the class.
# Perhaps we can overload the [] function?

probabilistic.model <- function(
  model
) {
  class(model) <- "probabilistic.model"
  return(model)
}

model.extract <- function(model, indices) {
  return(probabilistic.model(model$cpp[indices]))
}
