//
// Created by Dylan on 6/23/2022.
//

#include "cuDropout.cuh"
#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

namespace seblas {
    
    __global__ void dropoutD(Tensor* X, Tensor* Y, Tensor* mask, float p, long seed) {
        uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid >= X->dims.size) return;
        curandStateXORWOW_t state;
        curand_init(tid * seed, 0, 0, &state);
        float rand = curand_uniform(&state);
        if(rand > (1-p)) {
            Y->elements[tid] = X->elements[tid] / p;
            mask->elements[tid] = 1.0f;
        } else {
            Y->elements[tid] = 0.0f;
            mask->elements[tid] = 0.0f;
        }
    }
    
    __global__ void dropout4D(Tensor* X, Tensor* Y, Tensor* mask, float p, long seed) {
        uint32 tid = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        if(tid >= X->dims.size) return;
        
        float regisX[4] = {0};
        float regisY[4] = {0};
        float regisM[4] = {0};
        
        curandStateXORWOW_t state;
        curand_init(tid * seed, 0, 0, &state);
        regisM[0] = curand_uniform(&state) > (1-p) ? 1.0f : 0.0f;
        regisM[1] = curand_uniform(&state) > (1-p) ? 1.0f : 0.0f;
        regisM[2] = curand_uniform(&state) > (1-p) ? 1.0f : 0.0f;
        regisM[3] = curand_uniform(&state) > (1-p) ? 1.0f : 0.0f;
    
        toFloat4R(regisX[0]) = toFloat4R(X->elements[tid]);
        regisY[0] = regisM[0] * regisX[0] / p;
        regisY[1] = regisM[1] * regisX[1] / p;
        regisY[2] = regisM[2] * regisX[2] / p;
        regisY[3] = regisM[3] * regisX[3] / p;
        toFloat4R(Y->elements[tid]) = toFloat4R(regisY[0]);
        toFloat4R(mask->elements[tid]) = toFloat4R(regisM[0]);
    }
    
    __global__ void dropoutGradD(Tensor* dX, Tensor* dY, Tensor* mask, float p){
        uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid >= dX->dims.size) return;
        dX->elements[tid] += mask->elements[tid] * dY->elements[tid] * p;
    }
    
    __global__ void dropoutGrad4D(Tensor* dX, Tensor* dY, Tensor* mask, float p){
        uint32 tid = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        if(tid >= dX->dims.size) return;
        float regisDX[4] = {0};
        float regisDY[4] = {0};
        float regisM[4] = {0};
        toFloat4R(regisDY[0]) = toFloat4R(dY->elements[tid]);
        toFloat4R(regisDX[0]) = toFloat4R(dX->elements[tid]);
        toFloat4R(regisM[0]) = toFloat4R(mask->elements[tid]);
        regisDX[0] += regisM[0] * regisDY[0] * p;
        regisDX[1] += regisM[1] * regisDY[1] * p;
        regisDX[2] += regisM[2] * regisDY[2] * p;
        regisDX[3] += regisM[3] * regisDY[3] * p;
        toFloat4R(dX->elements[tid]) = toFloat4R(regisDX[0]);
    }
    
    Tensor* dropout(Tensor* X, Tensor* Y, Tensor* mask, float p) {
        long seed = (long)chrono::system_clock::now().time_since_epoch().count();
        
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (X->dims.size + block - 1) / block;
        
        if(X->dims.size %4 == 0){
            dropout4D<<<grid, block>>>(X, Y, mask, p, seed);
        } else {
            dropoutD<<<grid, block>>>(X, Y, mask, p, seed);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return Y;
    }
    
    Tensor* dropoutGrad(Tensor* dX, Tensor* dY, Tensor* mask, float p){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (dX->dims.size + block - 1) / block;
        if(dX->dims.size %4 == 0){
            dropoutGrad4D<<<grid, block>>>(dX, dY, mask, p);
        } else {
            dropoutGradD<<<grid, block>>>(dX, dY, mask, p);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return dX;
    }
} // seblas