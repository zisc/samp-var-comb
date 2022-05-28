dtdist <- function(x, dist, c, log = FALSE) {
  log.in <- log
  return(density(dist, x-c, log = log.in))
}

ptdist <- function(q, dist, c, log.p = FALSE) {
  return(distributional::cdf(dist, q-c, log = log.p))
}

qtdist <- function(p, dist, c, log.p = FALSE) {
  return(quantile(dist, p, log = log.p) + c)
}

rtdist <- function(n, dist, c) {
  return(generate(dist, n) + rep(c,n))
}

#dtdist <- function(x, dist, c, ..., log = FALSE)
