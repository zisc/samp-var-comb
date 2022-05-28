truncated_kernel_clt <- function(
  fit,
  dependent_index = -1
) {
  return(.Call(C_R_ManufactureTruncatedKernelCLT,
    fit,
    as.integer(dependent_index)
  ))
}