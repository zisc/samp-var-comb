CensoredLogScore <- function(
  open_lower_bound,
  closed_upper_bound,
  complement
) {
  open_lower_bound_coerced <- as.numeric(open_lower_bound)
  closed_upper_bound_coerced <- as.numeric(closed_upper_bound)
  if (open_lower_bound_coerced > closed_upper_bound_coerced) {
    stop("open_lower_bound > closed_upper_bound")
  }
  
  complement_coerced <- as.logical(complement)
  if (is.na(complement_coerced)) {
    stop("complement is missing")
  }
  
  cls <- .Call(
    C_R_ManufactureCensoredLogScore,
    open_lower_bound_coerced,
    closed_upper_bound_coerced,
    complement_coerced
  )
  attributes(cls) <- list(name = "Censored Log Score")
  
  return(cls)
}

ProbabilityCensoredLogScore <- function(
  open_lower_probability,
  closed_upper_probability,
  complement
) {
  open_lower_probability_coerced <- as.numeric(open_lower_probability)
  closed_upper_probability_coerced <- as.numeric(closed_upper_probability)
  if (open_lower_probability_coerced > closed_upper_probability_coerced) {
    stop("open_lower_probability > closed_upper_probability")
  }
  
  complement_coerced <- as.logical(complement)
  if (is.na(complement_coerced)) {
    stop("complement is missing")
  }
  
  cls <- .Call(
    C_R_ManufactureProbabilityCensoredLogScore,
    open_lower_probability_coerced,
    closed_upper_probability_coerced,
    complement_coerced
  )
  attributes(cls) <- list(name = "ProbabilityCensoredLogScore")
  
  return(cls)
}
