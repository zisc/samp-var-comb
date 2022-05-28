#include <sstream>
#include <stdexcept>
#include <vector>
#include <Rinternals.h>
#include <R_ext/Parse.h>
#include <R_protect_guard.hpp>
#include <R_support/function.hpp>

SEXP call_R_function(const char *fn_string, std::vector<SEXP>& args, R_protect_guard& protect_guard) {
    SEXP fn_string_sexp = protect_guard.protect(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(fn_string_sexp, 0, Rf_mkChar(fn_string));

    ParseStatus status;
    SEXP retrieve_fn_command = protect_guard.protect(R_ParseVector(fn_string_sexp, -1, &status, R_NilValue));
    if (status != PARSE_OK) {
        std::ostringstream ss;
        ss << "call_R_function: R_ParseVector status == " << status << " != PARSE_OK == " << PARSE_OK
           << ", fn_string == \"" << fn_string << "\".";
        throw std::logic_error(ss.str());
    }

    SEXP fn;
    for (int i = 0; i != Rf_length(retrieve_fn_command); ++i) {
        fn = Rf_eval(VECTOR_ELT(retrieve_fn_command, i), R_GlobalEnv);
    }

    SEXP call = protect_guard.protect(Rf_allocList(1 + args.size())); // "1 +" for the function fn_string.
    SEXP traverse = call;

    SET_TYPEOF(call, LANGSXP);
    SETCAR(traverse, fn);
    for (SEXP arg : args) {
        traverse = CDR(traverse);
        if (traverse == R_NilValue) {
            std::ostringstream ss;
            ss << "call_R_function: traverse == R_NilValue, args.size() == " << args.size()
               << ", Rf_length(call) == " << Rf_length(call) << '.';
            throw std::logic_error(ss.str());
        }
        SETCAR(traverse, arg);
    }

    return Rf_eval(call, R_GlobalEnv);
}

