TickScore <- function(probability) {
  probability_coerced <- as.numeric(probability)
  if (probability_coerced <= 0 || probability_coerced >= 1) {
    stop("probability <= 0 || probability >= 1")
  }
  ts <- .Call(
    C_R_ManufactureTickScore,
    probability_coerced
  )
  attributes(ts) <- list(name = "Tick Score")
  return(ts)
}
