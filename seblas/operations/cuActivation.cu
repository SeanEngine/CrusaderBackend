//
// Created by Dylan on 6/17/2022.
//

#include "cuActivation.cuh"

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define topOff(a,b) (a + b - 1)/(b)
#define sigmoidCalc(a) 1.0f/(1.0f + exp(-(a)))
#define tanhCalc(a) (exp(a) - exp(-(a)))/(exp(a) + exp(-(a)))

namespace seblas {
    
    __global__ void reluD(Tensor* input, Tensor* output){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            output->elements[idx] = max(input->elements[idx], 0.0f);
        }
    }
    
    __global__ void reluDGrad(Tensor* input, Tensor* output){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            output->elements[idx] = (input->elements[idx] > 0.0f) ? 1.0f : 0.0f;
        }
    }
    
    __global__ void relu4D(Tensor* input, Tensor* output){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = max(regis[0], 0.0f);
            regis[1] = max(regis[1],0.0f);
            regis[2] = max(regis[2],0.0f);
            regis[3] = max(regis[3],0.0f);
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void relu4DGrad(Tensor* input, Tensor* output){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] > 0.0f ? 1.0f : 0.0f;
            regis[1] = regis[1] > 0.0f ? 1.0f : 0.0f;
            regis[2] = regis[2] > 0.0f ? 1.0f : 0.0f;
            regis[3] = regis[3] > 0.0f ? 1.0f : 0.0f;
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void reluDGradFast(Tensor* Z, Tensor* dY, Tensor* dZ){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < Z->dims.size) {
            dZ->elements[idx] = (Z->elements[idx] > 0.0f) ? dY->elements[idx] : 0.0f;
        }
    }
    
    __global__ void relu4DGradFast(Tensor* Z, Tensor* dY, Tensor* dZ){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regisZ[4];
        float regisDY[4];
        if (idx < Z->dims.size) {
            toFloat4R(regisZ[0]) = toFloat4R(Z->elements[idx]);
            toFloat4R(regisDY[0]) = toFloat4R(dY->elements[idx]);
            regisDY[0] = regisZ[0] > 0.0f ? regisDY[0] : 0.0f;
            regisDY[1] = regisZ[1] > 0.0f ? regisDY[1] : 0.0f;
            regisDY[2] = regisZ[2] > 0.0f ? regisDY[2] : 0.0f;
            regisDY[3] = regisZ[3] > 0.0f ? regisDY[3] : 0.0f;
            toFloat4R(dZ->elements[idx]) = toFloat4R(regisDY[0]);
        }
    }
    
    __global__ void leakyReluD(Tensor* input, Tensor* output, float alpha){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            output->elements[idx] = val > 0 ? val : alpha * val;
        }
    }
    
    __global__ void leakyReluDGrad(Tensor* input, Tensor* output, float alpha){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            output->elements[idx] = val > 0 ? 1.0f : alpha;
        }
    }
    
    __global__ void leakyRelu4D(Tensor* input,Tensor* output, float alpha){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] > 0 ? regis[0] : alpha * regis[0];
            regis[1] = regis[1] > 0 ? regis[1] : alpha * regis[1];
            regis[2] = regis[2] > 0 ? regis[2] : alpha * regis[2];
            regis[3] = regis[3] > 0 ? regis[3] : alpha * regis[3];
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void leakyRelu4DGrad(Tensor* input, Tensor* output, float alpha){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] > 0 ? 1.0f : alpha;
            regis[1] = regis[1] > 0 ? 1.0f : alpha;
            regis[2] = regis[2] > 0 ? 1.0f : alpha;
            regis[3] = regis[3] > 0 ? 1.0f : alpha;
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void eluD(Tensor* input, Tensor* output, float alpha){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            output->elements[idx] = val < 0 ? alpha * (exp(val) - 1) : val;
        }
    }
    
    __global__ void elu4D(Tensor* input, Tensor* output, float alpha){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] < 0 ? alpha * (exp(regis[0]) - 1) : regis[0];
            regis[1] = regis[1] < 0 ? alpha * (exp(regis[1]) - 1) : regis[1];
            regis[2] = regis[2] < 0 ? alpha * (exp(regis[2]) - 1) : regis[2];
            regis[3] = regis[3] < 0 ? alpha * (exp(regis[3]) - 1) : regis[3];
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void eluDGrad(Tensor* input, Tensor* output, float alpha){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            output->elements[idx] = val < 0 ? alpha * exp(val) : 1.0f;
        }
    }
    
    __global__ void elu4DGrad(Tensor* input, Tensor* output, float alpha){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] < 0 ? alpha * exp(regis[0]) : 1.0f;
            regis[1] = regis[1] < 0 ? alpha * exp(regis[1]) : 1.0f;
            regis[2] = regis[2] < 0 ? alpha * exp(regis[2]) : 1.0f;
            regis[3] = regis[3] < 0 ? alpha * exp(regis[3]) : 1.0f;
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void sigmoidD(Tensor* input, Tensor* output){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            output->elements[idx] = sigmoidCalc(input->elements[idx]);
        }
    }
    
    __global__ void sigmoid4D(Tensor* input, Tensor* output){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = sigmoidCalc(regis[0]);
            regis[1] = sigmoidCalc(regis[1]);
            regis[2] = sigmoidCalc(regis[2]);
            regis[3] = sigmoidCalc(regis[3]);
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void sigmoidDGrad(Tensor* input, Tensor* output){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            output->elements[idx] = sigmoidCalc(val) * (1.0f - sigmoidCalc(val));
        }
    }
    
    __global__ void sigmoid4DGrad(Tensor* input, Tensor* output){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = sigmoidCalc(regis[0]) * (1.0f - sigmoidCalc(regis[0]));
            regis[1] = sigmoidCalc(regis[1]) * (1.0f - sigmoidCalc(regis[1]));
            regis[2] = sigmoidCalc(regis[2]) * (1.0f - sigmoidCalc(regis[2]));
            regis[3] = sigmoidCalc(regis[3]) * (1.0f - sigmoidCalc(regis[3]));
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void tanhD(Tensor* input, Tensor* output){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            output->elements[idx] = tanhCalc(input->elements[idx]);
        }
    }
    
    __global__ void tanh4D(Tensor* input, Tensor* output){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = tanhCalc(regis[0]);
            regis[1] = tanhCalc(regis[1]);
            regis[2] = tanhCalc(regis[2]);
            regis[3] = tanhCalc(regis[3]);
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    __global__ void tanhDGrad(Tensor* input, Tensor* output){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            output->elements[idx] = 1.0f - pow(tanhCalc(input->elements[idx]), 2.0f);
        }
    }
    
    __global__ void tanh4DGrad(Tensor* input, Tensor* output){
        uint32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = 1.0f - pow(tanhCalc(regis[0]), 2.0f);
            regis[1] = 1.0f - pow(tanhCalc(regis[1]), 2.0f);
            regis[2] = 1.0f - pow(tanhCalc(regis[2]), 2.0f);
            regis[3] = 1.0f - pow(tanhCalc(regis[3]), 2.0f);
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    
    Tensor* relu(Tensor* X, Tensor* outX){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            relu4D<<<grid, block>>>(X, outX);
        }else{
            reluD<<<grid, block>>>(X, outX);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* reluGrad(Tensor* X, Tensor* outX){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            relu4DGrad<<<grid, block>>>(X, outX);
        }else{
            reluDGrad<<<grid, block>>>(X, outX);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* reluGradFast(Tensor* Z, Tensor* dY, Tensor* dZ){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(Z->dims.size, block);
        if(Z->dims.size % 4 == 0){
            grid = topOff(Z->dims.size, block*4);
            relu4DGradFast<<<grid, block>>>(Z, dY, dZ);
        }else{
            reluDGradFast<<<grid, block>>>(Z, dY, dZ);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return dZ;
    }
    
    Tensor* lRelu(Tensor* X, Tensor* outX, float alpha){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            leakyRelu4D<<<grid, block>>>(X, outX, alpha);
        }else{
            leakyReluD<<<grid, block>>>(X, outX, alpha);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* lReluGrad(Tensor* X, Tensor* outX, float alpha){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            leakyRelu4DGrad<<<grid, block>>>(X, outX, alpha);
        }else{
            leakyReluDGrad<<<grid, block>>>(X, outX, alpha);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* elu(Tensor* X, Tensor* outX, float alpha){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            elu4D<<<grid, block>>>(X, outX, alpha);
        }else{
            eluD<<<grid, block>>>(X, outX, alpha);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* eluGrad(Tensor* X, Tensor* outX, float alpha){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            elu4DGrad<<<grid, block>>>(X, outX, alpha);
        }else{
            eluDGrad<<<grid, block>>>(X, outX, alpha);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* sigmoid(Tensor* X, Tensor* outX){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            sigmoid4D<<<grid, block>>>(X, outX);
        }else{
            sigmoidD<<<grid, block>>>(X, outX);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* sigmoidGrad(Tensor* X, Tensor* outX){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            sigmoid4DGrad<<<grid, block>>>(X, outX);
        }else{
            sigmoidDGrad<<<grid, block>>>(X, outX);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* tanh(Tensor* X, Tensor* outX){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            tanh4D<<<grid, block>>>(X, outX);
        }else{
            tanhD<<<grid, block>>>(X, outX);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
    
    Tensor* tanhGrad(Tensor* X, Tensor* outX){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = topOff(X->dims.size, block);
        if(X->dims.size % 4 == 0){
            grid = topOff(X->dims.size, block*4);
            tanh4DGrad<<<grid, block>>>(X, outX);
        }else{
            tanhDGrad<<<grid, block>>>(X, outX);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return outX;
    }
} // seblas