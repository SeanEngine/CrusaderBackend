//
// Created by Dylan on 6/17/2022.
//

#include "cuLoss.cuh"
#include "cuReduce.cuh"


namespace seblas {
    __global__ void crossEntropyPrepare(Parameter* Y, Tensor* labels){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx < Y->A->dims.size ){
            float labelVal = labels->elements[idx];
            labels->elements[idx] = -log(Y->A->elements[idx] + 1e-10f) * labelVal;
        }
    }
    
    float crossEntropyCalc(Parameter* Y, Tensor* buf){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (buf->dims.size + block - 1) / block;
        
        crossEntropyPrepare<<<grid, block>>>(Y, buf);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        
        float out =  reduce(buf, buf);
        buf->constFill(0.0f);
        return out;
    }
    
    void crossEntropyLoss(Parameter* Y, Tensor* labels){
        Y->A->copyToD2D(Y->dA);
        *Y->dA - labels;
    }
} // seblas