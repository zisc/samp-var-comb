seed_torch_rng <- function(seed) {
  invisible(.Call(C_R_seed_torch_rng, as.integer(seed)))
}

get_state_torch_rng <- function() {
  return(.Call(C_R_get_state_torch_rng))
}

set_state_torch_rng <- function(state) {
  invisible(.Call(C_R_set_state_torch_rng, as.raw(state)))
}
