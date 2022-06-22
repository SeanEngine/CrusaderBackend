//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_CUPOOL_CUH
#define CRUSADER_CUPOOL_CUH

#include "../tensor/Tensor.cuh"

namespace seblas {
    
    void maxPool(Tensor* X, Tensor* Y, Tensor* record);
    
    void maxPoolBack(Tensor* X, Tensor* Y, Tensor* record);
    
} // seblas

#endif //CRUSADER_CUPOOL_CUH
