//
// Created by Dylan on 7/5/2022.
//

#include "cuConcat.cuh"

namespace seblas {
    
    __global__ void concatD(Parameter** Xs, uint32 paramCount, Parameter* Y){
        uint32 id = blockIdx.x * blockDim.x + threadIdx.x;
        if(id >= Y->A->dims.size) return;
        uint32 n = Y->A->dims.n;
        uint32 optSize = Y->A->dims.size / n;
        uint32 cIndex = id % optSize;
        uint32 nIndex = id / optSize;
        
        Parameter* src;
        uint32 pId = 0;
    
        uint32 shift = 0;
        #pragma unroll
        while(shift + (Xs[pId]->A->dims.size / n) <= cIndex){
            shift += (Xs[pId]->A->dims.size / n);
            pId++;
            if(pId >= paramCount) return;
        }
        
        src = Xs[pId];
        uint32 srcIndex = nIndex * (src->A->dims.size / n) + (cIndex - shift);
        
        Y->A->elements[id] =  src->A->elements[srcIndex];
    }
    
    Parameter* concat(Parameter** Xs, uint32 paramCount, Parameter* Y){
        
        uint32 c = 0;
        for(uint32 id = 0; id < paramCount; id++){
            assert(Xs[id]->A->dims.n == Y->A->dims.n);
            assert(Xs[id]->A->dims.h == Y->A->dims.h);
            assert(Xs[id]->A->dims.w == Y->A->dims.w);
            c+= Xs[id]->A->dims.c;
        }
        assert(c == Y->A->dims.c);
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = (Y->A->dims.size + block - 1) / block;
        concatD<<<grid, block>>>(Xs, paramCount, Y);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return Y;
    }
    
    __global__ void concatGradD(Parameter* Y, Parameter** Xs, uint32 paramCount){
        uint32 id = blockIdx.x * blockDim.x + threadIdx.x;
        if(id >= Y->A->dims.size) return;
        uint32 n = Y->A->dims.n;
        uint32 optSize = Y->A->dims.size / n;
        uint32 cIndex = id % optSize;
        uint32 nIndex = id / optSize;
    
        Parameter* src;
        uint32 pId = 0;
    
        uint32 shift = 0;
        #pragma unroll
        while(shift + (Xs[pId]->A->dims.size / n) <= cIndex){
            shift += (Xs[pId]->A->dims.size / n);
            pId++;
            if(pId >= paramCount) return;
        }
    
        src = Xs[pId];
        uint32 srcIndex = nIndex * (src->A->dims.size / n) + (cIndex - shift);
    
        src->dAReserve->elements[srcIndex] += Y->dA->elements[id];
    }
    
    void concatGrads(Parameter* Y, Parameter** Xs, uint32 paramCount){
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = (Y->A->dims.size + block - 1) / block;
        concatGradD<<<grid, block>>>(Y, Xs, paramCount);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
} // seblas