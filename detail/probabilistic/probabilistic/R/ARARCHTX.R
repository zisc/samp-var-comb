ARARCHTX_tsibble_to_libtorch <- function(
  libtorch_data,
  tsibble_data,
  regressand,
  regressand_name,
  exogenous,
  exogenous_name
) {
  select <- dplyr::select
  all_of <- tidyselect::all_of
  key_vars <- tsibble::key_vars
  index_var <- tsibble::index_var
  setdiff <- generics::setdiff
  pivot_longer <- tidyr::pivot_longer
  
  tsibble_regressand <- tsibble_data %>%
    select(
      all_of(c(key_vars(tsibble_data), index_var(tsibble_data))),
      {{regressand}}
    )
  
  regressand_cols <- setdiff(
    colnames(tsibble_regressand),
    c(key_vars(tsibble_data), index_var(tsibble_data))
  )
  
  if (length(regressand_cols) > 1) {
    stop("Only one column can be passed as regressand in ARARCHTX().")
  }
  
  if (!(regressand_name %in% names(libtorch_data$tensor))) {
    libtorch_data <- libtorch_dict_append(
      libtorch_data,
      libtorch_dict(tsibble_regressand)
    )
  }
  
  if (!is.null(exogenous_name)) {
    tibble_exogenous <- tsibble_data %>%
      as_tibble() %>%
      select(
        all_of(c(key_vars(tsibble_data), index_var(tsibble_data))),
        {{exogenous}}
      ) %>%
      pivot_longer({{exogenous}}, names_to = "name", values_to = exogenous_name) %>%
      select(all_of(c(
        key_vars(tsibble_data),
        index_var(tsibble_data),
        "name",
        exogenous_name
      )))
    
    if (!(exogenous_name %in% names(libtorch_data$tensor))) {
      libtorch_data <- libtorch_dict_append(
        libtorch_data,
        libtorch_dict(tibble_with_indices(
          tibble_exogenous,
          key_vars(tsibble_data),
          index_var(tsibble_data),
          "name"
        ))
      )
    }
  }
  
  n.locations <- tsibble_regressand %>%
    as_tibble() %>%
    select(all_of(key_vars(tsibble_data))) %>%
    distinct() %>%
    nrow()
  
  return(list(
    n.locations = n.locations,
    libtorch_data = libtorch_data
  ))
}

ARARCHTX <- function(
  regressand,
  exogenous,
  regressand_name = deparse(substitute(regressand)),
  exogenous_name = if (base::missing(exogenous)) NULL else deparse(substitute(exogenous)),
  exogenous_enable = !base::missing(exogenous),
  pre_optimise = TRUE,
  ar_intercept = if (base::missing(exogenous)) "RandomEffect" else "Global",
  ar_intercept_init = NA_real_,
  ar_intercept_name = "ar_intercept",
  ar_intercept_soft_lower_bound = NA_real_,
  ar_intercept_soft_upper_bound = NA_real_,
  ar_intercept_scaling = 0.01,
  ar_intercept_barrier_scaling = 0.0,
  ar_intercept_enable = !is.null(ar_intercept) && ar_intercept %in% c("Global", "RandomEffect"),
  ar_exogenous_name = "ar_exogenous",
  ar_exogenous_init = NA_real_,
  ar_exogenous_soft_lower_bound = NA_real_,
  ar_exogenous_soft_upper_bound = NA_real_,
  ar_exogenous_scaling = 1.0,
  ar_exogenous_barrier_scaling = 0.0,
  ar_exogenous_enable = exogenous_enable,
  ar_coef_init = NA_real_,
  ar_coef_order = length(ar_coef_init),
  ar_coef_name = "ar_coefficients",
  ar_coef_soft_lower_bound = NA_real_,
  ar_coef_soft_upper_bound = NA_real_,
  ar_coef_scaling = 0.1,
  ar_coef_barrier_scaling = 0.0,
  ar_coef_enable = ar_coef_order >= 1,
  arch_intercept = if (base::missing(exogenous)) "RandomEffect" else "Global",
  arch_intercept_init = NA_real_,
  arch_intercept_name = "arch_intercept",
  arch_intercept_lower_bound = 0.0,
  arch_intercept_soft_upper_bound = 1000.0,
  arch_intercept_scaling = 0.01,
  arch_intercept_barrier_scaling = 1.0,
  arch_intercept_enable = !is.null(arch_intercept) && arch_intercept %in% c("Global", "RandomEffect"),
  arch_exogenous_name = "arch_exogenous",
  arch_exogenous_init = NA_real_,
  arch_exogenous_soft_lower_bound = NA_real_,
  arch_exogenous_soft_upper_bound = NA_real_,
  arch_exogenous_scaling = 1.0,
  arch_exogenous_barrier_scaling = 0.0,
  arch_exogenous_enable = exogenous_enable,
  arch_coef_init = NA_real_,
  arch_coef_order = length(arch_coef_init),
  arch_coef_name = "arch_coefficients",
  arch_coef_lower_bound = 0.0,
  arch_coef_soft_upper_bound = 1000.0,
  arch_coef_scaling = 0.001,
  arch_coef_barrier_scaling = 0.001,
  arch_coef_enable = arch_coef_order >= 1,
  var_transformation_crimp = NULL,
  var_transformation_catch = NULL,
  scoring_rule = LogScore(),
  learning_rate = 0.001,
  barrier_begin = 1e-3,
  barrier_end = 1e-9,
  barrier_decay = 0.5,
  tolerance_grad = 0.0,
  tolerance_change = 0.0,
  maximum_optimiser_iterations = as.integer(10000),
  optimiser_timeout_in_seconds = as.integer(1200)
) {
  select <- dplyr::select
  goup_by <- dplyr::group_by
  summarise <- dplyr::summarise
  
  regressand_name <- as.character(regressand_name)
  if (!is.null(exogenous_name)) exogenous_name <- as.character(exogenous_name)
  exogenous_enable <- as.logical(exogenous_enable)
  pre_optimise <- as.logical(pre_optimise)
  if (!is.null(ar_intercept)) ar_intercept <- as.character(ar_intercept)
  ar_intercept_init <- as.numeric(ar_intercept_init)
  ar_intercept_name <- as.character(ar_intercept_name)
  ar_intercept_soft_lower_bound <- as.numeric(ar_intercept_soft_lower_bound)
  ar_intercept_soft_upper_bound <- as.numeric(ar_intercept_soft_upper_bound)
  ar_intercept_scaling <- as.numeric(ar_intercept_scaling)
  ar_intercept_barrier_scaling <- as.numeric(ar_intercept_barrier_scaling)
  ar_intercept_enable <- as.logical(ar_intercept_enable)
  ar_exogenous_name <- as.character(ar_exogenous_name)
  ar_exogenous_init <- as.numeric(ar_exogenous_init)
  ar_exogenous_soft_lower_bound <- as.numeric(ar_exogenous_soft_lower_bound)
  ar_exogenous_soft_upper_bound <- as.numeric(ar_exogenous_soft_upper_bound)
  ar_exogenous_scaling <- as.numeric(ar_exogenous_scaling)
  ar_exogenous_barrier_scaling <- as.numeric(ar_exogenous_barrier_scaling)
  ar_coef_order <- as.integer(ar_coef_order)
  ar_coef_init <- as.numeric(ar_coef_init)
  ar_coef_soft_lower_bound <- as.numeric(ar_coef_soft_lower_bound)
  ar_coef_soft_upper_bound <- as.numeric(ar_coef_soft_upper_bound)
  ar_coef_name <- as.character(ar_coef_name)
  ar_coef_scaling <- as.numeric(ar_coef_scaling)
  ar_coef_barrier_scaling <- as.numeric(ar_coef_barrier_scaling)
  ar_coef_enable <- as.logical(ar_coef_enable)
  if (!is.null(arch_intercept)) arch_intercept <- as.character(arch_intercept)
  arch_intercept_init <- as.numeric(arch_intercept_init)
  arch_intercept_name <- as.character(arch_intercept_name)
  arch_intercept_lower_bound <- as.numeric(arch_intercept_lower_bound)
  arch_intercept_soft_upper_bound <- as.numeric(arch_intercept_soft_upper_bound)
  arch_intercept_scaling <- as.numeric(arch_intercept_scaling)
  arch_intercept_barrier_scaling <- as.numeric(arch_intercept_barrier_scaling)
  arch_intercept_enable <- as.logical(arch_intercept_enable)
  arch_exogenous_name <- as.character(arch_exogenous_name)
  arch_exogenous_init <- as.numeric(arch_exogenous_init)
  arch_exogenous_soft_lower_bound <- as.numeric(arch_exogenous_soft_lower_bound)
  arch_exogenous_soft_upper_bound <- as.numeric(arch_exogenous_soft_upper_bound)
  arch_exogenous_scaling <- as.numeric(arch_exogenous_scaling)
  arch_exogenous_barrier_scaling <- as.numeric(arch_exogenous_barrier_scaling)
  arch_exogenous_enable <- as.logical(arch_exogenous_enable)
  arch_coef_order <- as.integer(arch_coef_order)
  arch_coef_init <- as.numeric(arch_coef_init)
  arch_coef_name <- as.character(arch_coef_name)
  arch_coef_lower_bound <- as.numeric(arch_coef_lower_bound)
  arch_coef_soft_upper_bound <- as.numeric(arch_coef_soft_upper_bound)
  arch_coef_scaling <- as.numeric(arch_coef_scaling)
  arch_coef_barrier_scaling <- as.numeric(arch_coef_barrier_scaling)
  arch_coef_enable <- as.logical(arch_coef_enable)
  if (!is.null(var_transformation_crimp)) var_transformation_crimp <- as.numeric(var_transformation_crimp)
  if (!is.null(var_transformation_catch)) var_transformation_catch <- as.numeric(var_transformation_catch)
  learning_rate <- as.numeric(learning_rate)
  barrier_begin <- as.numeric(barrier_begin)
  barrier_end <- as.numeric(barrier_end)
  barrier_decay <- as.numeric(barrier_decay)
  tolerance_grad <- as.numeric(tolerance_grad)
  tolerance_change <- as.numeric(tolerance_change)
  maximum_optimiser_iterations <- as.integer(maximum_optimiser_iterations)
  optimiser_timeout_in_seconds <- as.integer(optimiser_timeout_in_seconds)
  
  if (!is.null(ar_intercept) && !(ar_intercept %in% c("Global", "RandomEffect"))) {
    stop("ar_intercept must be NULL, \"Global\" or \"RandomEffect\".")
  }
  
  if (!is.null(arch_intercept) && !(arch_intercept %in% c("Global", "RandomEffect"))) {
    stop("arch_intercept must be NULL, \"Global\" or \"RandomEffect\".")
  }
  
  if (ar_coef_order < 0) {
    stop("ar_order must be non-negative.")
  }
  
  if (arch_coef_order < 0) {
    stop("arch_order must be non-negative.")
  }
  
  untrained <- function(tsibble_data, libtorch_data = libtorch_dict()) {
    data_info <- ARARCHTX_tsibble_to_libtorch(
      libtorch_data,
      tsibble_data,
      {{regressand}},
      regressand_name,
      {{exogenous}},
      exogenous_name
    )
    libtorch_data <- data_info$libtorch_data
    tibble_data <- as_tibble(tsibble_data)
    
    pre_optimise_nested <- pre_optimise
    if (nrow(tsibble_data) <= 0) {
      pre_optimise_nested <- FALSE
    }
    
    num.exogenous.regressors <- NULL
    if (exogenous_enable) {
      num.exogenous.regressors <- tibble_data %>%
        select({{exogenous}}) %>%
        ncol()
    }
    
    ar_intercept_param <- NA_real_
    if (!is.null(ar_intercept)) {
      ar_intercept_param <- ar_intercept_init
      if (ar_intercept == "RandomEffect" && length(ar_intercept_param) == 1) {
        ar_intercept_param <- rep(ar_intercept_param, data_info$n.locations)
      }
    }
    
    ar_exogenous_param <- NA_real_
    if (ar_exogenous_enable) {
      ar_exogenous_param <- ar_intercept_init
      if (length(ar_intercept_init) == 1) {
        ar_exogenous_param <- rep(ar_exogenous_param, num.exogenous.regressors)
      }
    }
    
    ar_coef_param <- NA_real_
    if (ar_coef_enable) {
      ar_coef_param <- ar_coef_init
      if (length(ar_coef_init) == 1) {
        ar_coef_param <- rep(ar_coef_param, ar_coef_order)
      }
    }
    
    arch_intercept_param <- NA_real_
    if (!is.null(arch_intercept)) {
      arch_intercept_param <- arch_intercept_init
      if (arch_intercept == "RandomEffect" && length(arch_intercept_param) == 1) {
        arch_intercept_param <- rep(arch_intercept_param, data_info$n.locations)
      }
    }
    
    arch_exogenous_param <- NA_real_
    if (arch_exogenous_enable) {
      arch_exogenous_param <- arch_exogenous_init
      if (length(arch_exogenous_param) == 1) {
        arch_exogenous_param <- rep(arch_exogenous_param, num.exogenous.regressors)
      }
    }
    
    arch_coef_param <- NA_real_
    if (arch_coef_enable) {
      arch_coef_param <- arch_coef_init
      if (length(arch_coef_param) == 1) {
        arch_coef_param <- rep(arch_coef_param, arch_coef_order)
      }
    }
    
    parameter_guesses = list(list(
      ar_intercept = shapely.parameter(
        parameter = ar_intercept_param,
        lower_bound = ar_intercept_soft_lower_bound,
        upper_bound = ar_intercept_soft_upper_bound,
        parameter_scaling = ar_intercept_scaling,
        barrier_scaling = ar_intercept_barrier_scaling,
        enable = ar_intercept_enable
      ),
      ar_exogenous = shapely.parameter(
        parameter = ar_exogenous_param,
        lower_bound = ar_exogenous_soft_lower_bound,
        upper_bound = ar_exogenous_soft_upper_bound,
        parameter_scaling = ar_exogenous_scaling,
        barrier_scaling = ar_exogenous_barrier_scaling,
        enable = ar_exogenous_enable
      ),
      ar_coef = shapely.parameter(
        parameter = ar_coef_param,
        lower_bound = ar_coef_soft_lower_bound,
        upper_bound = ar_coef_soft_upper_bound,
        parameter_scaling = ar_coef_scaling,
        barrier_scaling = ar_coef_barrier_scaling,
        enable = ar_coef_enable
      ),
      arch_intercept = shapely.parameter(
        parameter = arch_intercept_param,
        lower_bound = arch_intercept_lower_bound,
        upper_bound = arch_intercept_soft_upper_bound,
        parameter_scaling = arch_intercept_scaling,
        barrier_scaling = arch_intercept_barrier_scaling,
        enable = arch_intercept_enable
      ),
      arch_exogenous = shapely.parameter(
        parameter = arch_exogenous_param,
        lower_bound = arch_exogenous_soft_lower_bound,
        upper_bound = arch_exogenous_soft_upper_bound,
        parameter_scaling = arch_exogenous_scaling,
        barrier_scaling = arch_exogenous_barrier_scaling,
        enable = arch_exogenous_enable
      ),
      arch_coef = shapely.parameter(
        parameter = arch_coef_param,
        lower_bound = arch_coef_lower_bound,
        upper_bound = arch_coef_soft_upper_bound,
        parameter_scaling = arch_coef_scaling,
        barrier_scaling = arch_coef_barrier_scaling,
        enable = arch_coef_enable
      )
    ))
    names(parameter_guesses[[1]]) <- c(
      ar_intercept_name,
      ar_exogenous_name,
      ar_coef_name,
      arch_intercept_name,
      arch_exogenous_name,
      arch_coef_name
    )
    
    if (is.null(var_transformation_crimp)) {
      var_transformation_crimp <- 0.08*(
        tibble_data %>%
          select(all_of(c(tsibble::key_vars(tsibble_data), tsibble::index_var(tsibble_data))), {{regressand}}) %>%
          group_by(across(tsibble::key_vars(tsibble_data))) %>%
          summarise(regressand_variance = var({{regressand}}), .groups = "drop") %>%
          select(regressand_variance) %>%
          summarise(regressand_variance = mean(regressand_variance))
      )[[1]]
    }
    
    if (is.null(var_transformation_catch)) {
      var_transformation_catch <- 0.001*var_transformation_crimp
    }
    
    buffers <- list(
      var_transformation_crimp,
      var_transformation_catch,
      regressand_name
    )
    if (!is.null(exogenous_name)) {
      buffers[[length(buffers)+1]] <- exogenous_name
    }
    
    return(new_libtorch_model_t(
      model = .Call(
        C_R_ManufactureARARCHTX,
        parameter_guesses,
        buffers,
        libtorch_data$dict,
        pre_optimise_nested
      )[[1]],
      name = NULL,
      success = NULL,
      optimiser_diagnostics = NULL,
      libtorch_data = libtorch_data
    ))
    
    # models <- list(
    #   models = .Call(C_R_ManufactureARARCHTX,
    #     parameter_guesses,
    #     buffers,
    #     libtorch_data$dict
    #   ),
    #   libtorch_data = libtorch_data
    # )
    # 
    # return(models)
  }
  
  train <- function(
    tsibble_data,
    libtorch_data = libtorch_dict(),
    scoring_rule_train = scoring_rule,
    learning_rate_train = learning_rate,
    barrier_begin_train = barrier_begin,
    barrier_end_train = barrier_end,
    barrier_decay_train = barrier_decay,
    tolerance_grad_train = tolerance_grad,
    tolerance_change_train = tolerance_change,
    maximum_optimiser_iterations_train = maximum_optimiser_iterations,
    optimiser_timeout_in_seconds_train = optimiser_timeout_in_seconds,
    return_diagnostics = TRUE
  ) {
    tg <- train_generic(
      untrained(tsibble_data, libtorch_data),
      paste0(
        "ARARCHTX(",
          deparse(substitute(regressand)), ", ",
          deparse(substitute(exogenous)), ", ",
          ar_coef_order, ", ",
          arch_coef_order,
        ")"
      ),
      scoring_rule_train,
      as.numeric(learning_rate_train),
      as.numeric(barrier_begin_train),
      as.numeric(barrier_end_train),
      as.numeric(barrier_decay_train),
      as.numeric(tolerance_grad_train),
      as.numeric(tolerance_change_train),
      as.integer(maximum_optimiser_iterations_train),
      as.integer(optimiser_timeout_in_seconds_train),
      return_diagnostics
    )
    libtorch_data <<- tg$libtorch_data
    return(tg)
  }
  
  retrain <- function(
    libtorch_model,
    libtorch_data = libtorch_model$libtorch_data,
    scoring_rule_retrain = scoring_rule,
    learning_rate_retrain = learning_rate,
    barrier_begin_retrain = barrier_begin,
    barrier_end_retrain = barrier_end,
    barrier_decay_retrain = barrier_decay,
    tolerance_grad_retrain = as.numeric(tolerance_grad),
    tolerance_change_retrain = as.numeric(tolerance_change),
    maximum_optimiser_iterations_retrain = maximum_optimiser_iterations,
    optimiser_timeout_in_seconds_retrain = optimiser_timeout_in_seconds,
    return_diagnostics = TRUE
  ) {
    tg <- train_generic(
      list(models = list(libtorch_model$model), libtorch_data = libtorch_data),
      libtorch_model$name,
      scoring_rule_retrain,
      learning_rate_retrain,
      barrier_begin_retrain,
      barrier_end_retrain,
      barrier_decay_retrain,
      tolerance_grad_retrain,
      tolerance_change_retrain,
      maximum_optimiser_iterations_retrain,
      optimiser_timeout_in_seconds_retrain,
      return_diagnostics
    )
    libtorch_data <<- tg$libtorch_data
    return(tg)
  }

  return(list(
    untrained = untrained,
    train = train,
    retrain = retrain
  ))
}
