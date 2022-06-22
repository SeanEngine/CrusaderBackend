//
// Created by Dylan on 6/17/2022.
//

#include "TensorUtils.cuh"
#include "Tensor.cuh"

#define topOff(a,b) (((a)+(b)-1)/(b))
#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

namespace seblas {
    __global__ void transposeD(Tensor* A, Tensor* B){
        uint32 row = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 col = blockIdx.y * blockDim.y + threadIdx.y;
        uint32 channel = blockIdx.z * blockDim.z + threadIdx.z;
        
        if(row < A->dims.h && col < A->dims.w && channel < A->dims.c * A->dims.n){
            B->elements[channel * A->dims.w * A->dims.h + col * A->dims.h + row]
                    = A->elements[channel * A->dims.w * A->dims.h + row * A->dims.w + col];
        }
    }
    
    __global__ void powD(Tensor* A, float val){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < A->dims.size){
            A->elements[idx] = pow(A->elements[idx], val);
        }
    }
    
    __global__ void pow4D(Tensor* A, float val){
        uint32 idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        float regis[4] = {0};
        if(idx < A->dims.size){
            toFloat4R(regis) = toFloat4R(A->elements[idx]);
            regis[0] = pow(regis[0], val);
            regis[1] = pow(regis[1], val);
            regis[2] = pow(regis[2], val);
            regis[3] = pow(regis[3], val);
            toFloat4R(A->elements[idx]) = toFloat4R(regis);
        }
    }
    
    Tensor* transpose(Tensor* A, Tensor* B){
        assert(A->dims.size == B->dims.size);
        dim3 block = CUDA_BLOCK_SIZE_3D;
        dim3 grid = {topOff(A->dims.w, block.x),
                     topOff(A->dims.h, block.y),
                     topOff(A->dims.c * A->dims.n, block.z)};
        transposeD<<<grid, block>>>(A, B);
        assertCuda(__FILE__, __LINE__);
        return B;
    }
    
    Tensor* transpose(Tensor* A){
        return transpose(A,A)->reshape(A->dims.n, A->dims.c, A->dims.w, A->dims.h);
    }
    
    Tensor* powTensor(Tensor* A, float val){
        assert(A->dims.size > 0);
        dim3 block = CUDA_BLOCK_SIZE;
        dim3 grid = {topOff(A->dims.size, block.x), 1, 1};
        
        if(A->dims.size %4 == 0){
            grid = {topOff(A->dims.size, block.x * 4), 1, 1};
            pow4D<<<grid, block>>>(A, val);
            assertCuda(__FILE__, __LINE__);
            return A;
        }
        
        powD<<<grid, block>>>(A, val);
        assertCuda(__FILE__, __LINE__);
        return A;
    }
} // seblas