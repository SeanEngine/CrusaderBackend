//
// Created by Dylan on 6/8/2022.
//

#ifndef CRUSADER_CULINEAR_CUH
#define CRUSADER_CULINEAR_CUH

//the computations for linear layers
//optimized for batchnorm and parallel calculations
#include "../tensor/Tensor.cuh"

namespace seblas {

    Tensor* linear(Tensor* X, Tensor* W, Tensor* B, Tensor* Y);
    
    Tensor* linearXGrad(Tensor* dY, Tensor* W, Tensor* dX);
    
    Tensor* linearParamGrad(Tensor* dY, Tensor* X, Tensor* dW, Tensor* dB);
    
} // seblas

#endif //CRUSADER_CULINEAR_CUH
