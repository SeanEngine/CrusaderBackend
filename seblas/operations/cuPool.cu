//
// Created by Dylan on 6/17/2022.
//

#include "cuPool.cuh"

namespace seblas {
    __global__ void maxPoolD(Tensor* X, Tensor* Y, Tensor* record, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW) {
        uint32 pid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32 nOffset = (Y->dims.c * Y->dims.h * Y->dims.w);
        const uint32 nDim = pid / nOffset;
        const uint32 cOffset = Y->dims.h * Y->dims.w;
        const uint32 cDim = (pid % nOffset) / cOffset;
        const uint32 hOffset = Y->dims.w;
        const uint32 hDim = ((pid % nOffset) % cOffset) / hOffset;
        const uint32 wDim = ((pid % nOffset) % cOffset) % hOffset;
        
        if(pid > Y->dims.size) return;
        
        float max = -1e15;
        uint32 indX = 0;
        uint32 indY = 0;
        #pragma unroll
        for(uint32 i = 0; i < rangeH; i++){
            #pragma unroll
            for(uint32 j = 0; j < rangeW; j++){
                float comp = X->elements[(nDim * nOffset + cDim * cOffset) * strideH * strideW
                                         + (hDim * strideH + i) * X->dims.w + wDim * strideW + j];
                if (comp > max){
                    indX = j;
                    indY = i;
                    max = comp;
                }
            }
        }
        
        record->elements[(nDim * nOffset + cDim * cOffset) * rangeH * rangeW
                         + (hDim * rangeH + indY) * X->dims.w + wDim * rangeW + indX] = 1;
        Y->elements[pid] = max;
    }
    
    __global__ void maxPoolBackD(Tensor* X, Tensor* Y, Tensor* record, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
        uint32 pid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32 nOffset = (Y->dims.c * Y->dims.h * Y->dims.w);
        const uint32 nDim = pid / nOffset;
        const uint32 cOffset = Y->dims.h * Y->dims.w;
        const uint32 cDim = (pid % nOffset) / cOffset;
        const uint32 hOffset = Y->dims.w;
        const uint32 hDim = ((pid % nOffset) % cOffset) / hOffset;
        const uint32 wDim = ((pid % nOffset) % cOffset) % hOffset;
        
        float max = -1e15;
        uint32 indX = 0;
        uint32 indY = 0;
        #pragma unroll
        for(uint32 i = 0; i < rangeH; i++){
            #pragma unroll
            for(uint32 j = 0; j < rangeW; j++){
                float comp = record->elements[(nDim * nOffset + cDim * cOffset) * rangeH * rangeW
                                              + (hDim * rangeH + i) * X->dims.w + wDim * rangeW + j];
                if (comp > max){
                    indX = j;
                    indY = i;
                    max = comp;
                }
            }
        }
        float grad = Y->elements[pid];
        X->elements[(nDim * nOffset + cDim * cOffset) * strideH * strideW
                    + (hDim * strideH + indY) * X->dims.w + wDim * strideW + indX] += grad;
        record->elements[(nDim * nOffset + cDim * cOffset) * strideH * strideW
                         + (hDim * strideH + indY) * X->dims.w + wDim * strideW + indX] = 0;
    }
    
    void maxPool(Tensor* X, Tensor* Y, Tensor* record, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
        assert((X->dims.h - (rangeH - strideH)) / strideH == Y->dims.h);
        assert((X->dims.w - (rangeW - strideW)) / strideW == Y->dims.w);
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = (Y->dims.size + block - 1)/block;
        maxPoolD<<<grid, block>>>(X, Y, record, strideH, strideW, rangeH, rangeW);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    void maxPoolBack(Tensor* X, Tensor* Y, Tensor* record, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = (Y->dims.size + block - 1)/block;
        maxPoolBackD<<<grid, block>>>(X, Y, record, strideH, strideW, rangeH, rangeW);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
} // seblas