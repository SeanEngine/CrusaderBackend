//
// Created by Dylan on 6/23/2022.
//

#ifndef CRUSADER_CUDROPOUT_CUH
#define CRUSADER_CUDROPOUT_CUH

#include "../tensor/Tensor.cuh"

namespace seblas {
    Tensor* dropout(Tensor* X, Tensor* Y, Tensor* mask, float p);
    
    Tensor* dropoutGrad(Tensor* dX, Tensor* dY, Tensor* mask, float p);
    
    Tensor* dropoutInfer(Tensor* X, Tensor* Y, float p);
} // seblas

#endif //CRUSADER_CUDROPOUT_CUH
