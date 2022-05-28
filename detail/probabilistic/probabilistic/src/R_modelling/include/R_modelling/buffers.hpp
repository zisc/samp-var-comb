#ifndef PROBABILISTIC_R_MODELLING_BUFFERS_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_BUFFERS_HPP_GUARD

#include <torch/torch.h>
#include <Rinternals.h>
#include <libtorch_support/Buffers.hpp>

Buffers to_buffers(SEXP buffers_R);

#endif

