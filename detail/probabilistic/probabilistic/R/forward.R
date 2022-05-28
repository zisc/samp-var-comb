forward <- function(object, ...) {
  UseMethod("forward")
}

forward.libtorch_model_t <- function(
  libtorch_model,
  observations_dict = libtorch_model$libtorch_data
) {
  pivot_longer <- tidyr::pivot_longer
  measured_vars <- measured_vars
  fable <- fabletools::fable
  key_vars <- key_vars
  index_var <- index_var
  
  forward_list <- .Call(C_R_forward,
    libtorch_model$model,
    observations_dict$dict
  )
  forward_tsibbles <- libtorch_list_to_tables(forward_list, observations_dict)
  return(lapply(forward_tsibbles, function(x) {
    x %>%
      as_tibble() %>%
      pivot_longer(
        cols = measured_vars(x),
        names_to = "process",
        values_to = "prob"
      ) %>%
      fable(
        key = all_of(c(key_vars(x))),
        index = all_of(index_var(x)),
        response = "process",
        distribution = "prob"
      )
  }))
}

# forward <- function(
#   models,
#   observations_dict
# ) {
#   nmodels <- length(models)
#   forwards <- vector("list", nmodels)
#   for (i in 1:nmodels) {
#     forward_list <- .Call(C_R_forward,
#       models[[i]],
#       observations_dict$dict
#     )
#     forward_tsibbles <- libtorch_list_to_tables(forward_list, observations_dict)
#     forwards[[i]] <- lapply(forward_tsibbles, function(x) {
#       x %>% 
#         as_tibble() %>%
#         pivot_longer(
#           cols = tsibble::measured_vars(x),
#           names_to = "process",
#           values_to = "prob"
#         ) %>%
#         fable(
#           key = all_of(c(tsibble::key_vars(x), "process")),
#           index = all_of(tsibble::index_var(x)),
#           response = "process",
#           distribution = "prob"
#         )
#     })
#   }
#   return(forwards)
# }
