//
// Created by Dylan on 6/17/2022.
//

#include "cuReduce.cuh"
#include "../tensor/TensorUtils.cuh"

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define topOff(a,b) (((a)+(b) - 1)/(b))

namespace seblas {
    
    //A warp reduction function that reduce values inside A given warp
    __device__ __forceinline__ float warpReduce(float val){
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, mask);
        }
        return val;
    }
    
    __device__ __forceinline__ float warpCompare(float val) {
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
            float temp = __shfl_xor_sync(0xffffffff, val, mask);
            val = temp >= val ? temp : val;
        }
        return val;
    }
    
    __device__ __forceinline__ float regisMax(const float* regis, uint32 count){
        float max = 0.0f;
        #pragma unroll
        for (uint32 i = 0; i < count; i++) {
            max = regis[i] > max ? regis[i] : max;
        }
        return max;
    }
    
    template <const uint32 BLOCK_WARPS>
    __global__ void reduceD(Tensor* A, Tensor* outA, uint32 reduceStep, uint32 procSize){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 reduceStepID = blockIdx.y;
        uint32 sourceStepID = blockIdx.z;
        uint32 tid = threadIdx.x;
        
        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float sum = idx < reduceStep && reduceStep * reduceStepID + idx < procSize
                    ? A->elements[sourceStepID * procSize + reduceStepID * reduceStep + idx] : 0;
        __syncthreads();
        
        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;
        
        __syncthreads();
        
        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) outA->elements[sourceStepID *
                                         topOff(procSize, reduceStep) + reduceStepID] = sum;
        }
    }
    
    template <const uint32 BLOCK_WARPS>
    __global__ void compareD(Tensor* A, Tensor* outA, uint32 reduceStep, uint32 procSize){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 reduceStepID = blockIdx.y;
        uint32 sourceStepID = blockIdx.z;
        uint32 tid = threadIdx.x;
        
        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float max = idx < reduceStep && reduceStep * reduceStepID + idx < procSize
                    ? A->elements[sourceStepID * procSize + reduceStepID * reduceStep + idx] : 0;
        __syncthreads();
        
        max = warpCompare(max);
        if(laneId==0) warpCache[warpId] = max;
        
        __syncthreads();
        
        if(warpId==0){
            max = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            max = warpCompare(max);
            if(laneId==0) outA->elements[sourceStepID *
                                         topOff(procSize, reduceStep) + reduceStepID] = max;
        }
    }
    
    //forward the softmax in the same block
    template <const uint32 BLOCK_WARPS>
    __global__ void softmaxD1024(Tensor* A, Tensor* out, uint32 step){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 stepID = blockIdx.y;
        uint32 tid = threadIdx.x;
        
        //warp compare :: find the max value in the current warp
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float max = tid < step ? A->elements[stepID * step + idx] : -1e15f;
        float val = tid < step ? max : 0.0f;
        __syncthreads();
        
        max = warpCompare(max);
        if(laneId==0) warpCache[warpId] = max;
        
        __syncthreads();
        
        if(warpId==0){
            max = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            max = warpCompare(max);
            if(laneId==0) warpCache[0] = max;
        }
        
        __syncthreads();
        
        //copy the max value to each thread
        max = warpCache[0];
        val = idx < step ? exp(val - max) : 0.0f;
        float sum = val;
        
        __syncthreads();
        
        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;
        
        __syncthreads();
        
        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) warpCache[0] = sum;
        }
        
        __syncthreads();
        
        val = val/(warpCache[0] + 1e-10);
        if(idx < step) out->elements[stepID * step + idx] = val;
    }
    
    template <const uint32 BLOCK_WARPS>
    __global__ void softmax4D4096(Tensor* A, Tensor* out, uint32 step){
        uint32 idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        
        uint32 stepID = blockIdx.y;
        uint32 tid = threadIdx.x;
        //warp compare :: find the max value in the current warp
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        
        float vals[4] = {0};
        if(idx < step) toFloat4R(vals) = toFloat4R(A->elements[stepID * step + idx]);
        float max = idx < step ?  regisMax(vals, 4) : -1e15f;
        __syncthreads();
        
        max = warpCompare(max);
        if(laneId==0) warpCache[warpId] = max;
        
        __syncthreads();
        
        if(warpId==0){
            max = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            max = warpCompare(max);
            if(laneId==0) warpCache[0] = max;
        }
        
        __syncthreads();
        
        //copy the max value to each thread
        max = warpCache[0];
        float sum = 0;
        
        if(idx < step) {
            #pragma unroll
            for (float &val: vals) {
                val = exp(val - max);
                sum += val;
            }
        }
        
        __syncthreads();
        
        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;
        
        __syncthreads();
        
        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) warpCache[0] = sum;
        }
        
        __syncthreads();
        
        #pragma unroll
        for(float & val : vals){
            val = val / (warpCache[0] + 1e-10f);
        }
        if(idx < step) toFloat4R(out->elements[stepID * step + idx]) = toFloat4R(vals);
    }
    
    //forward the exponentials
    __global__ void softmaxPrepare(Tensor* A, Tensor* buf, Tensor* out, uint32 step){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= step) return;
        uint32 stepID = blockIdx.y;
        
        float max = buf->elements[stepID];
        float val = A->elements[stepID * step + idx];
        float result = exp(val - max);
        out->elements[stepID * step + idx] = result;
    }
    
    //forward the division
    __global__ void softmaxFinalize(Tensor* buf, Tensor* out, uint32 step){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= step) return;
        uint32 stepID = blockIdx.y;
        
        float sum = buf->elements[stepID];
        float val = out->elements[stepID * step + idx];
        out->elements[stepID * step + idx] = val / sum;
    }
    
    //execution
    Tensor* reduce(Tensor* A, Tensor* out, Tensor* buffer, uint32 step){
        assert(A->dims.size % step == 0);
        assert(out->dims.size == A->dims.size / step);
        
        if(step < REDUCE_BLOCK + 1){
            dim3 grid = dim3(1, 1, A->dims.size / step);
            dim3 block = REDUCE_BLOCK;
            reduceD<REDUCE_WARPS><<<grid, block>>>(A, out, REDUCE_BLOCK, step);
            assertCuda(__FILE__, __LINE__);
            return out;
        }
        
        //step exceeds block capacity, using looped reduction
        assert(buffer != nullptr);
        assert(buffer->dims.size >= topOff(step, REDUCE_BLOCK) * A->dims.size / step);
        
        uint32 procSize = step;
        uint32 srcStepCount = A->dims.size / step;
        Tensor* src = A;
        
        while(procSize > 1){
            dim3 grid = dim3(1, topOff(procSize, REDUCE_BLOCK), srcStepCount);
            uint32 block = REDUCE_BLOCK;
            reduceD<REDUCE_BLOCK><<<grid, block>>>(src, buffer, REDUCE_BLOCK, procSize);
            assertCuda(__FILE__, __LINE__);
            src = buffer;
            procSize = topOff(procSize, REDUCE_BLOCK);
        }
        
        cudaMemcpy(out->elements, buffer->elements,
                   sizeof(float) * srcStepCount, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        buffer->constFill(0);
        return out;
    }
    
    float reduce(Tensor* A, Tensor* buffer){
        float output;
        auto* temp = Tensor::declare(1, 1)->instantiate();
        reduce(A, temp, buffer, A->dims.size);
        cudaMemcpy(&output, temp->elements, sizeof(float), cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        temp->eliminate();
        return output;
    }
    
    Tensor* rowReduce(Tensor* A, Tensor* out, Tensor* buffer){
        return reduce(A, out, buffer, A->dims.w);
    }
    
    Tensor* colReduce(Tensor* A, Tensor* out, Tensor* buffer){
        reduce(transpose(A), out, buffer, A->dims.w);
        transpose(A); //restore A
        return out;
    }
    
    Tensor* channelReduce(Tensor* A, Tensor* out, Tensor* buffer){
        return reduce(A, out, buffer, A->dims.h * A->dims.w);
    }
    
    
    Tensor* softmax(Tensor* A, Tensor* out, Tensor* buffer, uint32 step){
        assert(A->dims.size % step == 0);
        assert(out->dims.size == A->dims.size);
        
        /*
        if(step < REDUCE_BLOCK * 4 + 1 && step % 4 == 0){
            dim3 grid = dim3(1, A->dims.size / step);
            dim3 block = REDUCE_BLOCK;
            softmax4D4096<REDUCE_WARPS><<<grid, block>>>(A, out, step);
            assertCuda(__FILE__, __LINE__);
            return out;
        }*/
        
        if(step < REDUCE_BLOCK + 1){
            dim3 grid = dim3(1,  A->dims.size / step);
            dim3 block = REDUCE_BLOCK;
            softmaxD1024<REDUCE_WARPS><<<grid, block>>>(A, out, step);
            assertCuda(__FILE__, __LINE__);
            return out;
        }
        
        //step exceeds block capacity, using looped reduction
        assert(buffer != nullptr);
        assert(buffer->dims.size >= topOff(step, REDUCE_BLOCK) * A->dims.size / step);
        
        //find the max value for each step
        uint32 procSize = step;
        const uint32 srcStepCount = A->dims.size / step;
        Tensor* src = A;
        
        while(procSize > 1){
            dim3 grid = dim3(1, topOff(procSize, REDUCE_BLOCK), srcStepCount);
            uint32 block = REDUCE_BLOCK;
            compareD<REDUCE_BLOCK><<<grid, block>>>(src, buffer, REDUCE_BLOCK, procSize);
            assertCuda(__FILE__, __LINE__);
            src = buffer;
            procSize = topOff(procSize, REDUCE_BLOCK);
        }
        
        //do exp(A-max), prepare for reduction
        dim3 grid = dim3(topOff(step, REDUCE_BLOCK), srcStepCount);
        softmaxPrepare<<<grid, REDUCE_BLOCK>>>(A, buffer, out, step);
        assertCuda(__FILE__, __LINE__);
        
        //reduction
        procSize = step;
        src = out;
        
        while(procSize > 1){
            grid = dim3(1, topOff(procSize, REDUCE_BLOCK), srcStepCount);
            uint32 block = REDUCE_BLOCK;
            reduceD<REDUCE_BLOCK><<<grid, block>>>(src, buffer, REDUCE_BLOCK, procSize);
            assertCuda(__FILE__, __LINE__);
            src = buffer;
            procSize = topOff(procSize, REDUCE_BLOCK);
        }
        
        //apply softmax onto output
        grid = dim3(topOff(step, REDUCE_BLOCK), srcStepCount);
        softmaxFinalize<<<grid, REDUCE_BLOCK>>>(buffer, out, step);
        assertCuda(__FILE__, __LINE__);
        buffer->constFill(0);
        return out;
    }
    
    Tensor* rowSoftmax(Tensor* A, Tensor* out, Tensor* buffer){
        return softmax(A, out, buffer, A->dims.w);
    }
    
    Tensor* colSoftmax(Tensor* A, Tensor* out, Tensor* buffer){
        softmax(transpose(A), out, buffer, A->dims.w);
        transpose(A);
        return out;
    }
} // seblas