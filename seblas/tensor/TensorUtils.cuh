//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_TENSORUTILS_CUH
#define CRUSADER_TENSORUTILS_CUH

#include "Tensor.cuh"

namespace seblas {
    Tensor* transpose(Tensor* A);
    Tensor* transpose(Tensor* A, Tensor* B);
    Tensor* powTensor(Tensor* A, float val);
    
} // seblas

#endif //CRUSADER_TENSORUTILS_CUH
