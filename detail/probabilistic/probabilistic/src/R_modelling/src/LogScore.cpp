#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <modelling/score/LogScore.hpp>
#include <R_modelling/score/LogScore.hpp>

SEXP R_ManufactureLogScore(void) { return R_handle_exception([](){
    R_protect_guard protect_guard;
    return shared_ptr_to_EXTPTRSXP(
        ManufactureLogScore(),
        protect_guard
    );
});}

