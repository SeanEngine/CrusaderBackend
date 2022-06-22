//
// Created by Dylan on 6/18/2022.
//

#ifndef CRUSADER_CUBATCHNORM_CUH
#define CRUSADER_CUBATCHNORM_CUH


#include "../tensor/Tensor.cuh"

namespace seblas {
    Tensor* batchNorm(Tensor* X, Tensor* beta, Tensor* gamma,
                      Tensor* mean, Tensor* var, Tensor* Y);
    Tensor* batchNormGrad(Tensor* dY, Tensor* gamma, Tensor* X, Tensor* dX);
    
    void batchNormParamGrads(Tensor* dY, Tensor* dGamma, Tensor* dBeta,
                             Tensor* X);
}


#endif //CRUSADER_CUBATCHNORM_CUH
