forecast.libtorch_model_collection_t <- function(
  model_collection,
  point_forecast = list(.mean = mean),
  ...
) {
  measured_vars <- tsibble::measured_vars
  mutate <- dplyr::mutate
  full_join <- dplyr::full_join
  
  ret <- NULL
  for (m in model_collection$models) {
    fd <- forward(m$model, model_collection$libtorch_data)
    if (length(fd) != 1 || class(fd[[1]]) != "tbs_ts") {
      stop("forecast.libtorch_model_collection_t only works for models whos forecast returns a single tsibble. Consider using forward instead.")
    }
    ret_this <- fd[[1]]
    dependent_variable_name <- measured_vars(ret_this)
    if (length(dependent_variable_name) != 1) {
      stop("forecast.libtorch_model_collection_t only works for models which forecast a single dependent variable. Consider using forward instead.")
    }
    point_forecast_names = names(point_forecast)
    ret_this <- ret_this %>%
      mutate(.model = m$name)
    for (nm in point_forecast_names) {
      ret_this <- ret_this %>%
        mutate("{nm}" := point_forecast[[nm]](.data[[dependent_variable_name]]))
    }
    if (is.null(ret)) {
      ret <- ret_this
    } else {
      ret <- ret %>%
        full_join(
          ret_this,
          by = c(key_vars(ret), index_var(ret))
        )
    }
  }
}