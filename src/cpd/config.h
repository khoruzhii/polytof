// cpd/config.h
// Variant selection for CPD decomposition types.
// Define TOPP, BASE or COMM at compile time. Default: TOPP.

#pragma once

#include "core/paths.h"

#if !defined(TOPP) && !defined(BASE) && !defined(COMM)
  #define TOPP
#endif

#if defined(TOPP)
  #include "cpd/topp/scheme.h"
  #include "cpd/topp/tensor.h"
  namespace cpd {
    using Scheme = topp::Scheme;
    using Tensor = topp::Tensor;
    using topp::verify;
    using topp::load_tensor;
    using topp::trivial_decomposition;
    using topp::get_rank;
    using topp::get_rank_cubic;
    constexpr const char* SCHEMES_DIR = paths::CPD_TOPP_DIR;
  }
#elif defined(BASE)
  #include "cpd/base/scheme.h"
  #include "cpd/base/tensor.h"
  namespace cpd {
    using Scheme = base::Scheme;
    using Tensor = base::Tensor;
    using base::verify;
    using base::load_tensor;
    using base::trivial_decomposition;
    using base::get_rank;
    constexpr const char* SCHEMES_DIR = paths::CPD_BASE_DIR;
  }
#elif defined(COMM)
  #include "cpd/comm/scheme.h"
  #include "cpd/comm/tensor.h"
  namespace cpd {
    using Scheme = comm::Scheme;
    using Tensor = comm::Tensor;
    using comm::verify;
    using comm::load_tensor;
    using comm::trivial_decomposition;
    using comm::get_rank;
    constexpr const char* SCHEMES_DIR = paths::CPD_COMM_DIR;
  }
#endif