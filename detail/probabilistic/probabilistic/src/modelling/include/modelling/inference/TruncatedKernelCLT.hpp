#ifndef PROBABILISTIC_MODELLING_INFERENCE_TRUNCATED_KERNEL_CLT_HPP_GUARD
#define PROBABILISTIC_MODELLING_INFERENCE_TRUNCATED_KERNEL_CLT_HPP_GUARD

#include <memory>
#include <limits>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/inference/SamplingDistribution.hpp>

// This sampling distribution estimate uses standard results in
// extremum estimation. See, for example, Equation (3.6) of Andrews
// (2002) (Higher-order improvements of a computationally attractive
// k-step bootstrap for extremum estimators) where the estimator
// solves (3.5). k in the definition of W in equation (3.4) is chosen
// using the "Truncated Kernel" method outlined in Section 4.4 of
// Okui (2010) (Asymptotically unbiased estimation of autocovariances
// and autocorrelations with long panel data). Using the notation
// of Okui, we use S = sqrt(T), which satisfies the relevent
// assumptions of Okui's Theorem 6. On the other hand, if no time
// dimension is provided, this sampling distribution estimate
// assumes independence and we can take the equation in Andrews
// (2002) as given with k = 0 in Equation (3.4).
//
// For each tensor forecasted by fit.model (see the return type of
// ProbabilisticModule::forward), we permit temporal dependence
// to exist along a single dimension determined by dependent_index,
// but where observations that differ in any of the other indices
// are assumed independent. If dependent_index is non-negative, the
// dependent index is dependent_index. If dependent_index is negative,
// the dependent index is <tensor>.ndimension() + dependent_index.
// Note that zero based indexing is used throughout libtorch.


std::shared_ptr<SamplingDistribution> ManufactureTruncatedKernelCLT(
    std::shared_ptr<ProbabilisticModule> fit,
    int64_t dependent_index = std::numeric_limits<int64_t>::max()
);

#endif

