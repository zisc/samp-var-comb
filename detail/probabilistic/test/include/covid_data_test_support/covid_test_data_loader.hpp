#ifndef PROBABILISTIC_COVID_TEST_DATA_LOADER_HPP_GUARD
#define PROBABILISTIC_COVID_TEST_DATA_LOADER_HPP_GUARD

#include <R.h>
#include <Rinternals.h>
#undef ERROR
// If we don't undefine the ERROR macro after calling R headers,
// it is expanded somewhere in torch/torch.hpp causing this error:
// "
// /usr/share/R/include/R_ext/RS.h:55:17: error: expected unqualified-id before ‘)’ token
// #define ERROR   ),error(R_problem_buf);}
// "
// torch/torch.hpp is included in covid_data/structures.hpp, for example.
// Eventually we should create our own versions of R.h, Rinternals.h, etc.
// using #include_next that undefines this problematic macro anywhere it
// appears.
#include <covid_data/structures.hpp>

SEXP get_covid_test_data_R(void);
CovidData get_covid_test_data_libtorch(void);

#endif

