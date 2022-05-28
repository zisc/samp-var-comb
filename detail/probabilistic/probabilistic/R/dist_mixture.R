cpp_Mixture_to_dist_mixture <- function(weights, ...) {
  inner_join <- dplyr::inner_join
  
  component_lists <- list(...)
  common_series <- do.call(intersect, lapply(component_lists, function(x) { names(x) }))
  mixture_list <- vector("list", length(common_series))
  names(mixture_list) <- common_series
  
  for (series in common_series) {
    component_lists_1_series <- component_lists[[1]][[series]]
    idx_cols <- paste0("index_", 1:(length(component_lists_1_series)-1))
    names(component_lists_1_series) <- c(idx_cols, "comp_1")
    mixture_list_series_tibble <- as_tibble(component_lists_1_series)
    if (length(component_lists) > 1) {
      for (i in 2:length(component_lists)) {
        component_lists_i_series <- component_lists[[i]][[series]]
        if (length(component_lists_i_series) != length(idx_cols)+1) {
          stop("length(component_lists_i_series) != length(idx_cols)+1")
        }
        names(component_lists_i_series) <- c(idx_cols, paste0("comp_",i))
        mixture_list_series_tibble <- mixture_list_series_tibble %>%
          inner_join(as_tibble(component_lists_i_series), by = idx_cols)
      }
    }
    mixture_list_series <- as.list(mixture_list_series_tibble)
    names(mixture_list_series) <- NULL
    mixture_list_series_old_length <- length(mixture_list_series)
    mixture_list_series[[mixture_list_series_old_length+1]] <- do.call(
      function(...) { distributional::dist_mixture(..., weights = weights)},
      mixture_list_series[(length(idx_cols)+1):mixture_list_series_old_length]
    )
    mixture_list[[series]] <- mixture_list_series[c(1:length(idx_cols), length(mixture_list_series))]
  }
  
  return(mixture_list)
}
