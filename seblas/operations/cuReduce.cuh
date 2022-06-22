//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_CUREDUCE_CUH
#define CRUSADER_CUREDUCE_CUH

#include "../tensor/Tensor.cuh"

namespace seblas {
    const uint32 REDUCE_BLOCK = 1024;
    const uint32 REDUCE_WARPS = 32;
    
    /**
     * @brief Reduce the given tensor with steps.
     *
     * @param A input tensor
     * @param out output tensor with shape (A->size / steps, 1)
     * @param step the size of chunk to be summed up
     * @param buffer buffer for reduction (can be nullPointer)
     * @return
     */
    Tensor* reduce(Tensor* A, Tensor* out, Tensor* buffer, uint32 step);
    
    float reduce(Tensor* A, Tensor* buffer);
    
    //sum the elements in each row
    Tensor* rowReduce(Tensor* A, Tensor* out, Tensor* buffer);
    
    //sum the elements in each column
    Tensor* colReduce(Tensor* A, Tensor* out, Tensor* buffer);
    
    //sum the elements in each channel
    Tensor* channelReduce(Tensor* A, Tensor* out, Tensor* buffer);
    
    /**
     * @brief Do the softmax operations
     * @param A
     * @param out output tensor with shape (A->size / steps, 1)
     * @param step the size of chunk to forward softmax on
     * @param buffer buffer for reduction (can be nullPointer)
     * @return
     */
    Tensor* softmax(Tensor* A, Tensor* out, Tensor* buffer, uint32 step);
    
    //forward softmax on rows:
    Tensor* rowSoftmax(Tensor* A, Tensor* out, Tensor* buffer);
    
    //forward softmax on columns:
    Tensor* colSoftmax(Tensor* A, Tensor* out, Tensor* buffer);
    
} // seblas

#endif //CRUSADER_CUREDUCE_CUH
