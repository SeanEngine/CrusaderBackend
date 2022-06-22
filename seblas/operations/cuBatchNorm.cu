//
// Created by Dylan on 6/18/2022.
//

#include "cuBatchNorm.cuh"

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define BATCH_NORM_BLOCK 128
#define BATCH_NORM_MAX_PARALLEL 96

namespace seblas{
    
    template<const uint32 BLOCK, const uint32 MAX_PARALLEL>
    __global__ void batchNormD(Tensor* X, Tensor* beta, Tensor* gamma,
                               Tensor* mean, Tensor* var, Tensor* Y){
        const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= X->dims.size /X->dims.n ) return;
        
        //since each SM has access to 163KB of shared memory
        __shared__ float xs[BLOCK * MAX_PARALLEL];
        float meanVal = 0;
        float varVal = 0;
    
        #pragma unroll
        for(uint32 i = 0; i < MAX_PARALLEL; i++){
            xs[i * BLOCK + threadIdx.x] = 0;
        }
        __syncthreads();
        
        //load data into shared memory
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = X->elements[depth * (X->dims.size/X->dims.n) + idx];
            xs[threadIdx.x * MAX_PARALLEL + depth] = xVal;
            meanVal += xVal;
        }
        meanVal /= (float)X->dims.n;
        
        //compute variance
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[threadIdx.x * MAX_PARALLEL + depth];
            varVal += (xVal - meanVal) * (xVal - meanVal);
        }
        varVal /= ((float)X->dims.n);
        
        //compute xHat
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[threadIdx.x * MAX_PARALLEL + depth];
            float xHat = (xVal - meanVal) / (float)sqrt(varVal + 1e-8);
            float betaVal = beta->elements[idx];
            float gammaVal = gamma->elements[idx];
            Y->elements[depth * (Y->dims.size / Y->dims.n) + idx] = xHat * gammaVal + betaVal;
        }
        __syncthreads();
        
        mean->elements[idx] = meanVal;
        var->elements[idx] = varVal;
    }
    
    template<const uint32 BLOCK, const uint32 MAX_PARALLEL>
    __global__ void batchNormGradD(Tensor* dY, Tensor* gamma, Tensor* X, Tensor* dX){
        const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= dY->dims.size /dY->dims.n ) return;
        
        __shared__ float xs[BLOCK * MAX_PARALLEL];
        
        float meanVal = 0;
        float varVal = 0;
        float dVar = 0;
        float dMean = 0;
        float temp = 0;   // A part of dMean
    
        #pragma unroll
        for(uint32 i = 0; i < MAX_PARALLEL; i++){
            xs[i * BLOCK + threadIdx.x] = 0;
        }
        __syncthreads();
    
        //load data into shared memory
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = X->elements[depth * (X->dims.size/X->dims.n) + idx];
            xs[threadIdx.x * MAX_PARALLEL + depth] = xVal;
            meanVal += xVal;
        }
        meanVal /= (float)X->dims.n;
    
        //compute variance
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[threadIdx.x * MAX_PARALLEL + depth];
            varVal += (xVal - meanVal) * (xVal - meanVal);
        }
        varVal /= ((float)X->dims.n);
    
        //calculate dX
        #pragma unroll
        for(uint32 depth = 0; depth < dY->dims.n; depth++){
            float xVal =  xs[threadIdx.x * MAX_PARALLEL + depth];
            float gammaVal = gamma->elements[idx];
            
            //calculate dXHat
            float dxHat = dY->elements[depth * (dY->dims.size/dY->dims.n) + idx] * gammaVal;
            dVar += dxHat * (xVal - meanVal) * -0.5f * pow(varVal + 1e-8f, -3.0f/2.0f);
            dMean += dxHat * -1.0f / sqrt(varVal + 1e-8f);
            temp += -2.0f * (xVal - meanVal);
        }
        
        //calculate dMean
        dMean += dVar * temp / (float)dY->dims.n;
        
        //calculate dX
        #pragma unroll
        for(uint32 depth = 0; depth < dY->dims.n; depth++){
            float xVal = xs[threadIdx.x * MAX_PARALLEL + depth];
            float dy = dY->elements[depth * (dY->dims.size/dY->dims.n) + idx];
            float gammaVal = gamma->elements[idx];
            float dXHat = dy * gammaVal;
            dX->elements[depth * (dX->dims.size/dX->dims.n) + idx] = dXHat / sqrt(varVal + 1e-8f)
                      + dVar * (xVal - meanVal) * 2.0f / (float)dY->dims.n + dMean / (float)dY->dims.n;
        }
    }
    
    template<const uint32 BLOCK, const uint32 MAX_PARALLEL>
    __global__ void batchNormParamGradsD(Tensor* dY, Tensor* dGamma, Tensor* dBeta,
                                         Tensor* X){
        const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= dY->dims.size /dY->dims.n ) return;
        
        float dGammaVal = 0;
        float dBetaVal = 0;
        
        //since each SM has access to 163KB of shared memory
        __shared__ float xs[BLOCK * MAX_PARALLEL];
        float meanVal = 0;
        float varVal = 0;
    
        #pragma unroll
        for(uint32 i = 0; i < MAX_PARALLEL; i++){
            xs[i * BLOCK + threadIdx.x] = 0;
        }
        __syncthreads();
    
        //load data into shared memory
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = X->elements[depth * (X->dims.size/X->dims.n) + idx];
            xs[threadIdx.x * MAX_PARALLEL + depth] = xVal;
            meanVal += xVal;
        }
        meanVal /= (float)X->dims.n;
    
        //compute variance
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[threadIdx.x * MAX_PARALLEL + depth];
            varVal += (xVal - meanVal) * (xVal - meanVal);
        }
        varVal /= ((float)X->dims.n);
    
        //compute xHat
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[threadIdx.x * MAX_PARALLEL + depth];
            float xHat = (xVal - meanVal) / (float)sqrt(varVal + 1e-8);
            float dy = dY->elements[depth * (dY->dims.size/dY->dims.n) + idx];
            
            dBetaVal += dy;
            dGammaVal += dy * xHat;
        }
        
        dBeta->elements[idx] = dBetaVal;
        dGamma->elements[idx] = dGammaVal;
    }
    
    //[UNP] when batchsize < 2, results are different from cudnn
    Tensor* batchNorm(Tensor* X, Tensor* beta, Tensor* gamma,
                      Tensor* mean, Tensor* var, Tensor* Y){
        assert(X->dims.n == Y->dims.n);
        assert(X->dims.size == Y->dims.size);
        assert(X->dims.n <= BATCH_NORM_MAX_PARALLEL);
        
        uint32 block = BATCH_NORM_BLOCK;
        uint32 grid = ((X->dims.size / X->dims.n) + block - 1) / block;
        
        batchNormD<BATCH_NORM_BLOCK,BATCH_NORM_MAX_PARALLEL><<<grid, block>>>(X, beta, gamma, mean, var, Y);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return Y;
    }
    
    //[UNP]
    Tensor* batchNormGrad(Tensor* dY, Tensor* gamma, Tensor* X, Tensor* dX){
        assert(dY->dims.n == X->dims.n);
        assert(dY->dims.size == X->dims.size);
        
        uint32 block = BATCH_NORM_BLOCK;
        uint32 grid = ((X->dims.size / X->dims.n) + block - 1) / block;
        
        batchNormGradD<BATCH_NORM_BLOCK, BATCH_NORM_MAX_PARALLEL><<<grid, block>>>(dY, gamma, X, dX);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return dX;
    }
    
    void batchNormParamGrads(Tensor* dY, Tensor* dGamma, Tensor* dBeta,
                             Tensor* X){
        assert(dY->dims.n == X->dims.n);
        assert(dY->dims.size == X->dims.size);
        
        uint32 block = BATCH_NORM_BLOCK;
        uint32 grid = ((X->dims.size / X->dims.n) + block - 1) / block;
        
        batchNormParamGradsD<BATCH_NORM_BLOCK, BATCH_NORM_MAX_PARALLEL>
                <<<grid, block>>>(dY, dGamma, dBeta, X);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
}