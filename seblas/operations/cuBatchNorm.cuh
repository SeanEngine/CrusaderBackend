//
// Created by Dylan on 6/18/2022.
//

#ifndef CRUSADER_CUBATCHNORM_CUH
#define CRUSADER_CUBATCHNORM_CUH


#include "../tensor/Tensor.cuh"

namespace seblas {
    Tensor* batchNorm(Tensor* X, Tensor* beta, Tensor* gamma,
                      Tensor* mean, Tensor* var, Tensor* Y, Tensor* xHat);
    Tensor* batchNormGrad(Tensor* dY, Tensor* gamma, Tensor* X, Tensor* dX);
    
    void batchNormParamGrads(Tensor* dY, Tensor* xHat, Tensor* dGamma, Tensor* dBeta);
}


#endif //CRUSADER_CUBATCHNORM_CUH
