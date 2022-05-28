for_each_key <- function(object, ...) {
  UseMethod("for_each_key")
}

for_each_key.tbl_df <- function(x, keys, f) {
  select <- dplyr::select
  all_of <- dplyr::all_of
  distinct <- dplyr::distinct
  rowid_to_column <- tibble::rowid_to_column
  filter <- dplyr::filter
  select <- dplyr::select
  all_of <- dplyr::all_of
  distinct <- dplyr::distinct
  inner_join <- dplyr::inner_join
  
  keys_char <- sapply(keys, deparse)
  keys_df <- x %>% as_tibble %>% select(all_of(keys_char)) %>% distinct %>% rowid_to_column
  nkeys <- nrow(keys_df)
  out <- vector("list", nkeys)
  for (i in 1:nkeys) {
    keys_df_i <- keys_df %>% filter(rowid == i) %>% select(all_of(keys_char)) %>% distinct %>% rowid_to_column
    out[[i]] <- f(inner_join(x, keys_df_i, by = keys_char))
  }
  names(out) <- do.call(function(...) { paste(..., sep = "_") }, keys_df %>% select(all_of(keys_char)))
  return(out)
}

for_each_key.tbl_ts <- function(x, f) {
  for_each_key.tbl_df(x, key(x), f)
}

split_by_key <- function(object, ...) {
  UseMethod("split_by_key")
}

split_by_key.tbl_df <- function(x, keys) {
  for_each_key.tbl_df(x, keys, function(y){y})
}

split_by_key.tbl_ts <- function(x) {
  for_each_key.tbl_ts(x, function(y){y})
}

line_plot_each_key <- function(object, ...) {
  UseMethod("line_plot_each_key")
}

line_plot_each_key.tbl_df <- function(x, keys, mapping) {
  x_split <- split_by_key.tbl_df(x, keys)
  x_split_names <- names(x_split)
  
  plots <- vector("list", length(x_split))
  for (i in 1:length(x_split)) {
    plots[[i]] <- ggplot(data = x_split[[i]], mapping = mapping) +
      geom_line() +
      ggtitle(x_split_names[i])
  }
  names(plots) <- x_split_names
  
  return(plots)
}

line_plot_each_key.tbl_ts <- function(x, mapping) {
  line_plot_each_key.tbl_df(x, key(x), mapping)
}

line_plot_forecasts_each_key <- function(object, ...) {
  UseMethod("line_plot_forecasts_each_key")
}

line_plot_forecasts_each_key.fbl_ts <- function(forecasts, observations, mapping) {
  forecasts_split <- forecasts %>% split_by_key
  split_names <- names(forecasts_split)
  observations_split <- split_by_key(observations)[split_names]
  
  plots <- vector("list", length(forecasts_split))
  xlab_text <- substring(deparse(mapping[["x"]]), 2)
  ylab_text <- substring(deparse(mapping[["y"]]), 2)
  for (i in 1:length(forecasts_split)) {
    plots[[i]] <- ggplot() +
      autolayer(forecasts_split[[i]]) +
      geom_line(data = observations_split[[i]], mapping = mapping) +
      ggtitle(split_names[i]) +
      xlab(substring(deparse(mapping[["x"]]), 2)) +
      ylab(substring(deparse(mapping[["y"]]), 2))
  }
  names(plots) <- split_names
  
  return(plots)
}

save_plot_list <- function(
  plot_list,
  directory,
  device = "png",
  width = 25,
  height = 15,
  units = "cm",
  dpi = 300
) {
  if (!dir.exists(directory)) {
    dir.create(directory, recursive = TRUE)
  }
  
  names_plot_list <- names(plot_list)
  for (i in 1:length(plot_list)) {
    ggsave(
      paste0(names_plot_list[i], ".", device),
      plot = plot_list[[i]],
      device = device,
      path = directory,
      width = width,
      height = height,
      units = "cm"
    )
  }
}

optimiser_diagnostics_plot_list <- function(diagnostics) {
  mutate <- dplyr::mutate
  pivot_longer <- tidyr::pivot_longer
  
  plot_list_length <- 1 + length(diagnostics$paraameters) + length(diagnostics$gradient)
  plot_list <- vector("list", plot_list_length)
  plot_list_names <- vector("character", plot_list_length)
  
  j <- 1
  
  # Plot of scores.
  plot_list[[j]] <- ggplot(
    data = tibble(iteration = 1:length(diagnostics$score), score = diagnostics$score) %>%
      mutate(score_scaled = sign(score)*log10(1+abs(score))),
    mapping = aes(x = iteration, y = score_scaled)
  ) +
    geom_line() +
    ylab("score sign*log10(1+abs)")
  plot_list_names[j] <- "score"
  j <- j + 1
  
  # Plot of parameters.
  parameters_names <- names(diagnostics$parameters)
  for (i in 1:length(diagnostics$parameters)) {
    parameters_names_i <- parameters_names[i]
    parameters_i <- diagnostics$parameters[[i]]
    names(parameters_i) <- as.character(1:length(parameters_i))
    plot_frame <- as_tibble(parameters_i) %>%
      mutate(iteration = 1:length(diagnostics$score)) %>%
      pivot_longer(cols = as.character(1:length(parameters_i)), names_to = "idx", values_to = "value")
    plot_list[[j]] <- ggplot(plot_frame, aes(x = iteration, y = value, colour = idx)) +
      geom_line() +
      ylab(parameters_names_i) +
      theme(legend.position="none")
    plot_list_names[j] <- parameters_names_i
    j <- j + 1
  }
  
  # Plot of gradients.
  gradient_names <- names(diagnostics$gradient)
  for (i in 1:length(diagnostics$gradient)) {
    gradient_names_i <- gradient_names[i]
    gradient_i <- diagnostics$gradient[[i]]
    names(gradient_i) <- as.character(1:length(gradient_i))
    plot_frame <- as_tibble(gradient_i) %>%
      mutate(iteration = 1:length(diagnostics$score)) %>%
      pivot_longer(cols = as.character(1:length(gradient_i)), names_to = "idx", values_to = "value") %>%
      mutate(value_scaled = sign(value)*log10(1+abs(value)))
    plot_list[[j]] <- ggplot(plot_frame, aes(x = iteration, y = value_scaled, colour = idx)) +
      geom_line() +
      ylab(paste0(gradient_names_i, " gradient sign*log10(1+abs)")) +
      theme(legend.position="none")
    plot_list_names[j] <- gradient_names_i
    j <- j + 1
  }
  
  names(plot_list) <- plot_list_names
  
  return(plot_list)
}

#inference_plot_list <- function(
#  sampling_distribution,
#  monte_carlo_sample_size = 10000
#) {
#  parameter_draws <- parameters(draw_sampling_distribution(sampling_distribution, monte_carlo_sample_size))
#  performance_divergence_draws <- draw_performance_divergence(sampling_distribution, monte_carlo_sample_size)

inference_plot_list <- function(
  parameter_draws,
  performance_divergence_draws
) {
  plot_list <- list()
  plot_list_names <- NULL
  
  # Passing this to geom_boxplot() max the wiskers a 95% CI.
  boxplot_coef = qnorm(0.975)/(qnorm(0.75)-  qnorm(0.25)) - 0.5
  
  i <- 1
  
  # Performance Divergences.
  score_names <- names(performance_divergence_draws)
  for (j in 1:length(score_names)) {
    plot_list[[i]] <- ggplot(tibble("Score Divergence" = performance_divergence_draws[[j]]), aes(`Score Divergence`)) +
      geom_density() +
      ylab("Density")
    plot_list_names[i] <- score_names[j]
    i <- i + 1
  }
  
  # Parameter Draws
  nparams <- length(parameter_draws[[1]])
  parameter_names <- names(parameter_draws[[1]])
  for (j in 1:nparams) {
    param <- lapply(
      parameter_draws,
      function(x) {
        x[[j]]
      }
    )
    if (!is.null(param) && length(param) != 0) {
      param <- do.call(Map, c(f = c, param))
      param_dim <- length(param)
      for (k in 1:param_dim) {
        param_number <- ""
        if (param_dim > 1) { param_number <- paste0("_",as.character(k)) }
        param_name <- paste0(parameter_names[j], param_number)
        plot_list[[i]] <- ggplot(
          data = tibble(Parameter = param_name, Value = param[[k]]),
          mapping = aes(y = Value, x = Parameter)
        ) +
          geom_boxplot(coef = boxplot_coef, outlier.shape = NA)
        plot_list_names[i] <- param_name
        i <- i + 1
      }
    }
  }
  
  names(plot_list) <- plot_list_names
  
  return(plot_list)
}
