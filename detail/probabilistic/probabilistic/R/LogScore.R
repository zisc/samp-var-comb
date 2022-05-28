LogScore <- function() {
  ls <- .Call(C_R_ManufactureLogScore)
  attributes(ls) <- list(name = "Log Score")
  return(ls)
}