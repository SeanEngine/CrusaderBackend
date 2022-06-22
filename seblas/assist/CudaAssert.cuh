//
// Created by Dylan on 6/2/2022.
//

#ifndef CRUSADER_CUDAASSERT_CUH
#define CRUSADER_CUDAASSERT_CUH

#include "../../seio/logging/LogUtils.cuh"

using namespace seio;
namespace seblas {

    /**
     * @brief Asserts that CUDA device has no errors
     *
     * @param file file containing code
     */
    void assertCuda(const char* file, int line);

} // seblas

#endif //CRUSADER_CUDAASSERT_CUH
