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
                    + (hDim * strideH + indY) * X->dims.w + wDim * strideW + indX] = grad;
        record->elements[(nDim * nOffset + cDim * cOffset) * strideH * strideW
                         + (hDim * strideH + indY) * X->dims.w + wDim * strideW + indX] = 0;
    }
    
    __global__ void avgPoolD(Tensor* X, Tensor* Y, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW) {
        uint32 pid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32 nOffset = (Y->dims.c * Y->dims.h * Y->dims.w);
        const uint32 nDim = pid / nOffset;
        const uint32 cOffset = Y->dims.h * Y->dims.w;
        const uint32 cDim = (pid % nOffset) / cOffset;
        const uint32 hOffset = Y->dims.w;
        const uint32 hDim = ((pid % nOffset) % cOffset) / hOffset;
        const uint32 wDim = ((pid % nOffset) % cOffset) % hOffset;
        
        if(pid > Y->dims.size) return;
        
        float sum = 0;
        #pragma unroll
        for(uint32 i = 0; i < rangeH; i++){
            #pragma unroll
            for(uint32 j = 0; j < rangeW; j++){
                sum += X->elements[(nDim * nOffset + cDim * cOffset) * strideH * strideW
                                         + (hDim * strideH + i) * X->dims.w + wDim * strideW + j];
            }
        }
        
        Y->elements[pid] = sum / (float)(rangeH * rangeW);
    }
    
    __global__ void avgPoolBackD(Tensor* X, Tensor* Y, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
        uint32 pid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32 nOffset = (Y->dims.c * Y->dims.h * Y->dims.w);
        const uint32 nDim = pid / nOffset;
        const uint32 cOffset = Y->dims.h * Y->dims.w;
        const uint32 cDim = (pid % nOffset) / cOffset;
        const uint32 hOffset = Y->dims.w;
        const uint32 hDim = ((pid % nOffset) % cOffset) / hOffset;
        const uint32 wDim = ((pid % nOffset) % cOffset) % hOffset;
        
        float grad = Y->elements[pid];
        #pragma unroll
        for(uint32 i = 0; i < rangeH; i++){
            #pragma unroll
            for(uint32 j = 0; j < rangeW; j++){
                X->elements[(nDim * nOffset + cDim * cOffset) * strideH * strideW
                            + (hDim * strideH + i) * X->dims.w + wDim * strideW + j] = grad / (float)(rangeH * rangeW);
            }
        }
    }
    
    __global__ void globalAvgPoolBackD(Tensor* dX, Tensor* dY){
        uint32 tid = threadIdx.x + blockDim.x * blockIdx.x;
        uint32 step = dX->dims.h * dX->dims.w;
        if(tid > dY->dims.c * dY->dims.n) return;
        
        float grad = dY->elements[tid];
        grad /= (float)step;
    
        #pragma unroll
        for(uint32 i = 0; i < step; i++){
            dX->elements[tid * step + i] = grad;
        }
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
    
    void avgPool(Tensor* X, Tensor* Y, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
        assert((X->dims.h - (rangeH - strideH)) / strideH == Y->dims.h);
        assert((X->dims.w - (rangeW - strideW)) / strideW == Y->dims.w);
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = (Y->dims.size + block - 1)/block;
        avgPoolD<<<grid, block>>>(X, Y, strideH, strideW, rangeH, rangeW);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    void avgPoolBack(Tensor* X, Tensor* Y, uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = (Y->dims.size + block - 1)/block;
        avgPoolBackD<<<grid, block>>>(X, Y, strideH, strideW, rangeH, rangeW);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    void globalAvgPool(Tensor* X, Tensor* Y, Tensor* buffer){
        uint32 step = X->dims.h * X->dims.w;
        reduce(X, Y, buffer, step);
        *Y / (float)step;
        assertCuda(__FILE__, __LINE__);
    }
    
    void globalAvgPoolBack(Tensor* dX, Tensor* dY){
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = (dY->dims.size + block - 1)/block;
        globalAvgPoolBackD<<<grid, block>>>(dX, dY);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
} // seblas