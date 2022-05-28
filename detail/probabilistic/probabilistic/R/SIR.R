SIR_tsibble_to_libtorch <- function(
  epidemic_libtorch,
  epidemic_tsibble,
  cumulative_cases,
  population,
  panel_covariates,
  locational_covariates,
  panel_covariates_name = "panel_covariates",
  locational_covariates_name = "locational_covariates"
) {
  select <- dplyr::select
  all_of <- dplyr::all_of
  key_vars <- tsibble::key_vars
  index_var <- tsibble::index_var
  count <- dplyr::count
  summarise <- dplyr::summarise
  distinct <- dplyr::distinct
  setdiff <- generics::setdiff
  across <- dplyr::across
  pivot_longer <- tidyr::pivot_longer
  
  epidemic_tsibble_cumulative_cases <- epidemic_tsibble %>%
    select(
      all_of(c(tsibble::key_vars(epidemic_tsibble), tsibble::index_var(epidemic_tsibble))),
      {{cumulative_cases}}
    )
  
  cumulative_cases_name <- setdiff(
    colnames(epidemic_tsibble_cumulative_cases),
    c(tsibble::key_vars(epidemic_tsibble), tsibble::index_var(epidemic_tsibble))
  )
  
  if (length(cumulative_cases_name) > 1) {
    stop("Only one column can be passed as cumulative cases in SIR().")
  }
  
  if (!(cumulative_cases_name %in% names(epidemic_libtorch$tensor))) {
    epidemic_libtorch <- libtorch_dict_append(
      epidemic_libtorch,
      libtorch_dict(epidemic_tsibble_cumulative_cases)
    )
  }
  
  epidemic_tsibble_population <- epidemic_tsibble %>%
    as_tibble() %>%
    select(all_of(tsibble::key_vars(epidemic_tsibble)), {{population}}) %>%
    distinct()
  
  population_name <- setdiff(
    colnames(epidemic_tsibble_population),
    c(tsibble::key_vars(epidemic_tsibble), tsibble::index_var(epidemic_tsibble))
  )
  
  if (length(population_name) > 1) {
    stop("Only one column can be passed as population in SIR().")
  }
  
  max_population_repeats <- (epidemic_tsibble_population %>%
    select(all_of(tsibble::key_vars(epidemic_tsibble))) %>%
    count(across(all_of(tsibble::key_vars(epidemic_tsibble)))) %>%
    summarise(n = max(n)))$n
  
  if (max_population_repeats != 1) {
    stop("In SIR(), the population column must not vary by the tsibble index.")
  }
  
  if (!(population_name %in% names(epidemic_libtorch$tensor))) {
    epidemic_libtorch <- libtorch_dict_append(
      epidemic_libtorch,
      libtorch_dict(tibble_with_indices(
        epidemic_tsibble_population,
        tsibble::key_vars(epidemic_tsibble)
      ))
    )
  }
  
  epidemic_tsibble_panel_covariates <- epidemic_tsibble %>%
    as_tibble() %>%
    select(
      all_of(c(tsibble::key_vars(epidemic_tsibble), tsibble::index_var(epidemic_tsibble))),
      {{panel_covariates}}
    ) %>%
    pivot_longer({{panel_covariates}}, names_to = "name", values_to = panel_covariates_name) %>%
    select(all_of(c(
      tsibble::key_vars(epidemic_tsibble),
      tsibble::index_var(epidemic_tsibble),
      "name",
      panel_covariates_name
    )))
  
  if (!(panel_covariates_name %in% names(epidemic_libtorch$tensor))) {
    epidemic_libtorch <- libtorch_dict_append(
      epidemic_libtorch,
      libtorch_dict(tibble_with_indices(
        epidemic_tsibble_panel_covariates,
        tsibble::key_vars(epidemic_tsibble),
        tsibble::index_var(epidemic_tsibble),
        "name"
      ))
    )
  }
  
  epidemic_tsibble_locational_covariates <- epidemic_tsibble %>%
    as_tibble() %>%
    select(
      all_of(tsibble::key_vars(epidemic_tsibble)),
      {{locational_covariates}}
    ) %>%
    distinct()
  
  max_locational_covariates_repeats <- (epidemic_tsibble_locational_covariates %>%
    select(all_of(tsibble::key_vars(epidemic_tsibble))) %>%
    count(across(all_of(tsibble::key_vars(epidemic_tsibble)))) %>%
    summarise(n = max(n)))$n
  
  if (max_locational_covariates_repeats != 1) {
    stop("In SIR(), the locational_covariates columns must not vary by the tsibble index.")
  }
  
  epidemic_tsibble_locational_covariates <- epidemic_tsibble_locational_covariates %>%
    pivot_longer({{locational_covariates}}, names_to = "name", values_to = locational_covariates_name) %>%
    select(all_of(c(
      tsibble::key_vars(epidemic_tsibble),
      "name",
      locational_covariates_name
    )))
  
  if (!(locational_covariates_name %in% names(epidemic_libtorch$tensor))) {
    epidemic_libtorch <- libtorch_dict_append(
      epidemic_libtorch,
      libtorch_dict(tibble_with_indices(
        epidemic_tsibble_locational_covariates,
        tsibble::key_vars(epidemic_tsibble),
        "name"
      ))
    )
  }
  
  n.locations <- epidemic_tsibble_population %>%
    select(all_of(tsibble::key_vars(epidemic_tsibble))) %>%
    distinct() %>%
    nrow()
  
  n.panel.covariates <- epidemic_tsibble_panel_covariates %>%
    select(name) %>%
    distinct() %>%
    nrow()
  
  n.locational.covariates <- epidemic_tsibble_locational_covariates %>%
    select(name) %>%
    distinct() %>%
    nrow()
  
  return(list(
    n.locations = n.locations,
    n.panel.covariates = n.panel.covariates,
    n.locational.covariates = n.locational.covariates,
    cumulative_cases_name = cumulative_cases_name,
    population_name = population_name,
    panel_covariates_name = panel_covariates_name,
    locational_covariates_name = locational_covariates_name,
    libtorch_data = epidemic_libtorch
  ))
}

SIR <- function(
  cumulative_cases,
  population,
  panel_covariates,
  locational_covariates,
  panel_covariates_name = "panel_covariates",
  locational_covariates_name = "locational_covariates",
  alpha_use_locationals = TRUE,
  beta_use_locationals = FALSE,
  beta_use_panels = TRUE,
  beta_use_cases = TRUE,
  use_arch = TRUE,
  theta_cases_order = 7,
  arch_order = 7,
  alpha_name = "alpha",
  alpha_guess = distributional::dist_normal(0.5, 0.1),
  alpha_lower_bound = 0.05,
  alpha_upper_bound = 1.0,
  alpha_scaling = 0.5,
  alpha_barrier_scaling = 1.0,
  phi_name = "phi",
  phi_guess = distributional::dist_normal(0, 0.4),
  phi_scaling = 1.0,
  theta_intercept_name = "theta_intercept",
  theta_intercept_guess = distributional::dist_normal(0, 0.3),
  theta_intercept_soft_lower_bound = -5.0,
  theta_intercept_soft_upper_bound = 5.0,
  theta_intercept_scaling = 0.5,
  theta_locational_name = "theta_locational",
  theta_locational_guess = theta_intercept_guess,
  theta_locational_soft_lower_bound = theta_intercept_soft_lower_bound,
  theta_locational_soft_upper_bound = theta_intercept_soft_upper_bound,
  theta_locational_scaling = 2*theta_intercept_scaling,
  theta_panel_name = "theta_panel",
  theta_panel_guess = theta_locational_guess,
  theta_panel_soft_lower_bound = theta_locational_soft_lower_bound,
  theta_panel_soft_upper_bound = theta_locational_soft_upper_bound,
  theta_panel_scaling = theta_locational_scaling,
  theta_cases_name = "theta_cases",
  # theta_cases_guess = distributional::dist_normal(0, 0.2),
  theta_cases_guess = distributional::dist_degenerate(0),
  theta_cases_soft_lower_bound = -2.0,
  theta_cases_soft_upper_bound = 2.0,
  theta_cases_scaling = 10,
  epsilon_name = "epsilon",
  epsilon_guess = distributional::dist_normal(0, 0.3),
  epsilon_soft_lower_bound = -5.0,
  epsilon_soft_upper_bound = 5.0,
  epsilon_scaling = 0.3,
  gamma_name = "gamma",
  gamma_guess = distributional::dist_beta(2,6),
  gamma_lower_bound = 0.0,
  gamma_upper_bound = 1.0,
  gamma_scaling = 1.0,
  gamma_barrier_scaling = 0.001,
  sigma2_name = "sigma2",
  sigma2_guess = distributional::dist_exponential(100),
  sigma2_lower_bound = 0.0,
  sigma2_soft_upper_bound = 0.1,
  sigma2_scaling = 200.0,
  sigma2_barrier_scaling = 0.05,
  arch_name = "arch",
  arch_guess = distributional::dist_degenerate(0.01),
  arch_lower_bound = 0.0,
  arch_soft_upper_bound = 1.0,
  arch_scaling = 10.0,
  arch_barrier_scaling = 0.05,
  nguesses = 100,
  scoring_rule = LogScore(),
  learning_rate = 0.001,
  barrier_begin = 5.0,
  barrier_end = 1e-3,
  barrier_decay = 0.95,
  maximum_optimiser_iterations = as.integer(10000),
  optimiser_timeout_in_seconds = as.integer(300)
) {
  if (arch_order < 1) {
    arch_order <- 1
    use_arch <- FALSE
  }
  
  untrained <- function(epidemic_tsibble, epidemic_libtorch = libtorch_dict(), nmodels = nguesses) {
    data_info <- SIR_tsibble_to_libtorch(
      epidemic_libtorch,
      epidemic_tsibble,
      {{cumulative_cases}},
      {{population}},
      {{panel_covariates}},
      {{locational_covariates}},
      panel_covariates_name,
      locational_covariates_name
    )
    epidemic_libtorch <- data_info$libtorch_data
    
    parameter_guesses = vector("list", nmodels)
    for (i in 1:nmodels) {
      parameter_guesses[[i]] <- list(
        alpha = shapely.parameter(
          parameter = generate(alpha_guess, data_info$n.locations)[[1]],
          lower_bound = alpha_lower_bound,
          upper_bound = alpha_upper_bound,
          parameter_scaling = alpha_scaling,
          barrier_scaling = alpha_barrier_scaling
        ),
        phi = shapely.parameter(
          parameter = generate(phi_guess, data_info$n.locational.covariates)[[1]],
          parameter_scaling = phi_scaling,
          enable = alpha_use_locationals
        ),
        theta_intercept = shapely.parameter(
          parameter = generate(theta_intercept_guess, 1)[[1]],
          lower_bound = theta_intercept_soft_lower_bound,
          upper_bound = theta_intercept_soft_upper_bound,
          parameter_scaling = theta_intercept_scaling,
          enable = beta_use_locationals
        ),
        theta_locational = shapely.parameter(
          parameter = generate(theta_locational_guess, data_info$n.locational.covariates)[[1]],
          lower_bound = theta_locational_soft_lower_bound,
          upper_bound = theta_locational_soft_upper_bound,
          parameter_scaling = theta_locational_scaling,
          enable = beta_use_locationals
        ),
        theta_panel = shapely.parameter(
          parameter = generate(theta_panel_guess, data_info$n.panel.covariates)[[1]],
          lower_bound = theta_panel_soft_lower_bound,
          upper_bound = theta_panel_soft_upper_bound,
          parameter_scaling = theta_panel_scaling,
          enable = beta_use_panels
        ),
        theta_cases = shapely.parameter(
          parameter = generate(theta_cases_guess, theta_cases_order)[[1]],
          lower_bound = theta_cases_soft_lower_bound,
          upper_bound = theta_cases_soft_upper_bound,
          parameter_scaling = theta_cases_scaling,
          enable = beta_use_cases
        ),
        epsilon = shapely.parameter(
          parameter = generate(epsilon_guess, data_info$n.locations)[[1]],
          lower_bound = epsilon_soft_lower_bound,
          upper_bound = epsilon_soft_upper_bound,
          parameter_scaling = epsilon_scaling
        ),
        gamma = shapely.parameter(
          parameter = generate(gamma_guess, 1)[[1]],
          lower_bound = gamma_lower_bound,
          upper_bound = gamma_upper_bound,
          parameter_scaling = gamma_scaling,
          barrier_scaling = gamma_barrier_scaling
        ),
        sigma2 = shapely.parameter(
          parameter = generate(sigma2_guess, data_info$n.locations)[[1]],
          lower_bound = sigma2_lower_bound,
          upper_bound = sigma2_soft_upper_bound,
          parameter_scaling = sigma2_scaling,
          barrier_scaling = sigma2_barrier_scaling
        ),
        arch = shapely.parameter(
          parameter = generate(arch_guess, arch_order)[[1]],
          lower_bound = arch_lower_bound,
          upper_bound = arch_soft_upper_bound,
          parameter_scaling = arch_scaling,
          barrier_scaling = arch_barrier_scaling,
          enable = use_arch
        )
      )
      names(parameter_guesses[[i]]) <- c(
        alpha_name,
        phi_name,
        theta_intercept_name,
        theta_locational_name,
        theta_panel_name,
        theta_cases_name,
        epsilon_name,
        gamma_name,
        sigma2_name,
        arch_name
      )
    }
    
    models <- list(
      models = .Call(C_R_ManufactureSIR,
        parameter_guesses,
        list(
          data_info$cumulative_cases_name,
          data_info$population_name,
          data_info$panel_covariates_name,
          data_info$locational_covariates_name
        ),
        epidemic_libtorch$dict
      ),
      libtorch_data = epidemic_libtorch
    )
    return(models)
  }
  
  train <- function(epidemic_tsibble, epidemic_libtorch = libtorch_dict()) {
    return(train_generic(
      untrained(epidemic_tsibble, epidemic_libtorch, nmodels = nguesses),
      paste0(
        "SIR(",
        deparse(substitute(cumulative_cases)), ", ",
        deparse(substitute(population)), ", ",
        deparse(substitute(panel_covariates)), ", ",
        deparse(substitute(locational_covariates)),
        ")"
      ),
      scoring_rule,
      learning_rate,
      barrier_begin,
      barrier_end,
      barrier_decay,
      maximum_optimiser_iterations,
      optimiser_timeout_in_seconds
    ))
  }
  
  return(list(
    untrained = untrained,
    train = train
  ))
}

# Problem with integrating with fable: the "The training function" section of
# https://fabletools.tidyverts.org/articles/extension_models.html reads
# "The .data argument is a single series tsibble (no keys)..."
# But this is a panel data set, not a purely time series one, so we cannot
# go SIR(covid, cumulative_cases ~ ...) as we would like. instead, we will
# write our own model and forecast functions.
# specials_sir <- new_specials(
#   population = function(...) {
#     pop <- select(self$data, ...)
#     if (ncol(pop) > 1) {
#       stop("Population must refer to one column only in SIR()")
#     }
#     return(pop)
#   },
#   panel_covariates = function(...) {
#     select(self$data, ...)
#   },
#   locational_covariates = function(...) {
#     loc <- self$data %>%
#       as_tibble %>%
#       select(all_of(key_vars(self$data)), ...) %>%
#       distinct()
#     if (loc %>% select(all_of(key_vars(self$data))) %>% n_distinct() < nrow(loc)) {
#       stop("Locational covariates must vary by location only, and not by time, in SIR()")
#     }
#     return(loc)
#   },
#   xreg = function(...) {
#     stop("Exogenous regressors aren't supported by SIR()")
#   }
# )
# 
# train_sir <- function(.data, specials, ...) {
#   mv <- tsibble::measured_vars(.data)
#   if (length(mv) > 1) {
#     stop("Response is cumulative cases, and must be univariates.")
#   }
#   
# }
