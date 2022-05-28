# Not in use, see comment at the top of probabilistic_model.R

probabilistic.score <- function(
  cpp.score
) {
  ret <- list("cpp.score" = cpp.score)
  class(ret) <- "probabilistic.score"
  return(ret)
}