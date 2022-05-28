#include <exception>
#include <boost/thread.hpp>
#include <boost/log/trivial.hpp>
#include <R_ext/Print.h>
#include <R_ext/Rdynload.h>
#include <boost_log_R/sink_backend.hpp>
#include <dll_visibility.h>
#include <log/trivial.hpp>
#include <create_tensor.hpp>
#include <R_data_translation/R_list_to_libtorch_dict.hpp>
#include <R_data_translation/libtorch_dict_to_R_list.hpp>
#include <R_modelling/torch_rng.hpp>
#include <R_modelling/functional/empirical_coverage.hpp>
#include <R_modelling/model/ARARCHTX.hpp>
#include <R_modelling/model/Ensemble.hpp>
#include <R_modelling/model/serialise.hpp>
#include <R_modelling/model/parameters.hpp>
#include <R_modelling/model/average_score.hpp>
#include <R_modelling/model/draw_observations.hpp>
#include <R_modelling/score/LogScore.hpp>
#include <R_modelling/score/CensoredLogScore.hpp>
#include <R_modelling/score/TickScore.hpp>
#include <R_modelling/inference/sampling_distribution_draws.hpp>
#include <R_modelling/inference/performance_divergence_draws.hpp>
#include <R_modelling/inference/TruncatedKernelCLT.hpp>
#include <R_modelling/fit.hpp>
#include <R_modelling/forward.hpp>

extern "C" DLL_PUBLIC void R_init_libprobabilistic(DllInfo*);
void R_init_libprobabilistic_impl(DllInfo*);

extern "C" DLL_PUBLIC void R_init_libprobabilistic(DllInfo *dll_info) {
    try {
        R_init_libprobabilistic_impl(dll_info);
    } catch (std::exception& e) {
        REprintf("probabilistic initialisation failed with C++ exception: %s\n", e.what());
    } catch (...) {
        REprintf("probabilistic initialisation failed with unknown C++ exception.\n");
    }
}

void R_init_libprobabilistic_impl(DllInfo *dll_info) {
    // Initialise libraries here.

    initialise_boost_log_R_sink_backend();

    {
        auto num_cpu_cores = boost::thread::physical_concurrency();
        if (num_cpu_cores) { // If num_cpu_cores != 0 (i.e. is known), set pytorch threads to match.
            torch::set_num_threads(num_cpu_cores);
        }
    }

    // Register native routines here.    
    static const R_CallMethodDef callMethods[] = {
        {"libtorch_test_create_tensor", (DL_FUNC) &libtorch_test_create_tensor, 0},
        {"R_new_libtorch_dict", (DL_FUNC) &R_new_libtorch_dict, 0},
        {"R_seed_torch_rng", (DL_FUNC) &R_seed_torch_rng, 1},
        {"R_get_state_torch_rng", (DL_FUNC) &R_get_state_torch_rng, 0},
        {"R_set_state_torch_rng", (DL_FUNC) &R_set_state_torch_rng, 1},
        {"R_libtorch_dict_append_list", (DL_FUNC) &R_libtorch_dict_append_list, 3},
        {"R_libtorch_dict_append_dict", (DL_FUNC) &R_libtorch_dict_append_dict, 2},
        {"R_libtorch_dict_combine", (DL_FUNC) &R_libtorch_dict_combine, 2},
        {"R_libtorch_dict_time_slice", (DL_FUNC) &R_libtorch_dict_time_slice, 2},
        {"R_libtorch_dict_to_list", (DL_FUNC) &R_libtorch_dict_to_list, 1},
        {"R_ManufactureARARCHTX", (DL_FUNC) &R_ManufactureARARCHTX, 4},
        {"R_ManufactureEnsemble", (DL_FUNC) &R_ManufactureEnsemble, 3},
        {"R_change_components", (DL_FUNC) &R_change_components, 2},
        {"R_serialise_model", (DL_FUNC) &R_serialise_model, 1},
        {"R_deserialise_model", (DL_FUNC) &R_deserialise_model, 2},
        {"R_ManufactureLogScore", (DL_FUNC) &R_ManufactureLogScore, 0},
        {"R_ManufactureCensoredLogScore", (DL_FUNC) &R_ManufactureCensoredLogScore, 3},
        {"R_ManufactureProbabilityCensoredLogScore", (DL_FUNC) &R_ManufactureCensoredLogScore, 3},
        {"R_ManufactureTickScore", (DL_FUNC) &R_ManufactureTickScore, 1},
        {"R_forward", (DL_FUNC) &R_forward, 2},
        {"R_fit", (DL_FUNC) &R_fit, 12},
        {"R_parameters", (DL_FUNC) &R_parameters, 1},
        {"R_change_parameters", (DL_FUNC) &R_change_parameters, 2},
        {"R_average_score", (DL_FUNC) &R_average_score, 3},
        {"R_average_score_out_of_sample", (DL_FUNC) &R_average_score_out_of_sample, 4},
        {"R_draw_observations", (DL_FUNC) &R_draw_observations, 3},
        {"R_sampling_distribution_draws", (DL_FUNC) &R_sampling_distribution_draws, 2},
        {"R_performance_divergence_draws", (DL_FUNC) &R_performance_divergence_draws, 3},
        {"R_ManufactureTruncatedKernelCLT", (DL_FUNC) &R_ManufactureTruncatedKernelCLT, 2},
        {"R_empirical_coverage", (DL_FUNC) &R_empirical_coverage, 6},
        {"R_empirical_coverage_expanding_window_obs", (DL_FUNC) &R_empirical_coverage_expanding_window_obs, 6},
        {"R_empirical_coverage_expanding_window_noobs", (DL_FUNC) &R_empirical_coverage_expanding_window_noobs, 5},
        {nullptr, nullptr, 0}
    };
    
    R_registerRoutines(
        dll_info,
        nullptr,        // .C
        callMethods,    // .Call
        nullptr,        // .Fortran
        nullptr         // .External
    );

    // Ensure that this dll is not searched for
    // entry points specified by character strings,
    // and that only registered routines in this
    // dll may be called.
    R_useDynamicSymbols(dll_info, FALSE);
    R_forceSymbols(dll_info, TRUE);
}

