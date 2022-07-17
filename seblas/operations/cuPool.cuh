//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_CUPOOL_CUH
#define CRUSADER_CUPOOL_CUH

#include "../tensor/Tensor.cuh"
#include "../operations/cuReduce.cuh"

namespace seblas {
    
    void maxPool(Tensor* X, Tensor* Y, Tensor* record,
                 uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW);
    
    void maxPoolBack(Tensor* dX, Tensor* dY, Tensor* record,
                     uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW);
    
    void avgPool(Tensor* X, Tensor* Y, uint32 strideH, uint32 strideW, uint32 rangeH
                 , uint32 rangeW);
    
    void avgPoolBack(Tensor* dX, Tensor* dY, uint32 strideH, uint32 strideW, uint32 rangeH
                     , uint32 rangeW);
    
    void globalAvgPool(Tensor* X, Tensor* Y, Tensor* buffer);
    
    void globalAvgPoolBack(Tensor* dX, Tensor* dY);
} // seblas

#endif //CRUSADER_CUPOOL_CUH
