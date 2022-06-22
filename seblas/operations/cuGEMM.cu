//
// Created by Dylan on 6/3/2022.
//

#include "cuGEMM.cuh"
#include "../assist/Inspections.cuh"
//DO NOT MODIFY THESE CONSTANTS
//UNLESS YOU ARE AWARE OF WHAT THEY MEANS
#define BM 128
#define BN 128
#define BK 8
#define RM 8
#define RN 8

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
namespace seblas {

/**
 * same as the previous gemmPrefetching4NN but do not use float4 in global memory loading
 * to support all matrix dimensions
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 *
 * [Unit Test Passed]
 */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetchingNN(Tensor *A, Tensor *B, Tensor *C){
        const uint32 M = A->dims.h;
        const uint32 N = B->dims.w;
        const uint32 K = A->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_K;
        const int readThreadPerRowB = BLOCK_N;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        //This outer loop is special designed for support of batch normalization
        //allowing linear layers to run samples parallel to each others
        #pragma unroll
        for(uint32 nDim = 0; nDim < B->dims.n; nDim++) {
            uint32 nShift = nDim * B->dims.h * B->dims.w;
            float *ptrA = A->elements + blockIdx.y * BLOCK_M * K;
            float *ptrB = nShift + B->elements + blockIdx.x * BLOCK_N;
            
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                if (blockM + readRowA + i < M && readColA < K) {
                    tileA[0][readColA][readRowA + i] = ptrA[(readRowA + i) * K + readColA];
                }else{
                    tileA[0][readColA][readRowA + i] = 0;
                }
            }
            
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                if (readRowB + i < K && blockN + readColB < N) {
                    tileB[0][readRowB + i][readColB] = ptrB[(readRowB + i) * N + readColB];
                }else{
                    tileB[0][readRowB + i][readColB] = 0;
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
            }
            
            ///main loop
            int writeStageFlag = 1;
            #pragma unroll
            for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
                //prefetch
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA;
                        bufferA[loadIndex] = blockM + readRowA + i < M && readColA + nextTileID < K ?
                                             ptrA[(readRowA + i) * K + readColA + nextTileID] : 0;
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB;
                        bufferB[loadIndex] = readRowB + i + nextTileID < K && blockN + readColB < N ?
                                             ptrB[(readRowB + i + nextTileID) * N + readColB] : 0;
                    }
                }
                
                int nextStageFlag = writeStageFlag ^ 1;
                
                //compute the part that is already in the registers and load the next segment
                #pragma unroll
                for (int i = 0; i < BLOCK_K - 1; i++) {
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm += 4) {
                        toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                                tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                    }
                    
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn += 4) {
                        toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                                tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                    }
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm++) {
                        #pragma unroll
                        for (int rn = 0; rn < REGIS_N; rn++) {
                            regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                        }
                    }
                }
                
                //load the data in the register buffers to tiles
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA;
                        tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB;
                        tileB[writeStageFlag][readRowB + i][readColB] = bufferB[loadIndex];
                    }
                    
                    __syncthreads();
                    writeStageFlag ^= 1;  //switch
                }
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[0][rm]) = toFloat4R(
                            tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[0][rn]) = toFloat4R(
                            tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                    }
                }
            }
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                        C->elements[nDim * C->dims.h * C->dims.w + (blockM + threadIdx.y * REGIS_M + rm) * N
                                    + blockN + threadIdx.x * REGIS_N + rn] = regisC[rm][rn];
                        regisC[rm][rn] = 0;
                    }
                }
            }
        }
    }
    
    
    
    /**
    * The first matrix is transposed before computation
    * @tparam BLOCK_M
    * @tparam BLOCK_N
    * @tparam BLOCK_K
    * @tparam REGIS_M
    * @tparam REGIS_N
    * @param A
    * @param B
    * @param C
     *
     * [Unit Test Passed]
    */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetchingTN(Tensor *A, Tensor *B, Tensor *C){
        const uint32 M = A->dims.w;
        const uint32 N = B->dims.w;
        const uint32 K = A->dims.h;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float* ptrA = A->elements;
        float* ptrB = B->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_M;
        const int readThreadPerRowB = BLOCK_N;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        //This outer loop is special designed for support of batch normalization
        //allowing linear layers to run samples parallel to each others
        #pragma unroll
        for(uint32 nDim = 0; nDim < B->dims.n; nDim++) {
            uint32 nShift = nDim * B->dims.h * B->dims.w;
            
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                if (readRowA + i < K && blockM + readColA < M) {
                    //The mat A is not transposed since it will be transposed in smem
                    tileA[0][readRowA + i][readColA] = ptrA[(readRowA + i) * M + blockM + readColA];
                } else {
                    tileA[0][readRowA + i][readColA] = 0;
                }
            }
            
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                if (readRowB + i < K && blockN + readColB < N) {
                    tileB[0][readRowB + i][readColB] = ptrB[nShift + (readRowB + i) * N + blockN + readColB];
                } else {
                    tileB[0][readRowB + i][readColB] = 0;
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
            }
            
            
            ///main loop
            int writeStageFlag = 1;
            #pragma unroll
            for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
                //prefetch
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA;
                        //here the mat A is automatially transposed while reading
                        bufferA[loadIndex] = readRowA + i + nextTileID < K && blockM + readColA < M ?
                                             ptrA[(readRowA + i + nextTileID) * M + blockM + readColA] : 0;
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB;
                        bufferB[loadIndex] = readRowB + i + nextTileID < K && blockN + readColB < N ?
                                             ptrB[nShift + (readRowB + i + nextTileID) * N + blockN + readColB] : 0;
                    }
                }
                
                int nextStageFlag = writeStageFlag ^ 1;
                
                //compute the part that is already in the registers and load the next segment
                #pragma unroll
                for (int i = 0; i < BLOCK_K - 1; i++) {
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm += 4) {
                        toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                                tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                    }
                    
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn += 4) {
                        toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                                tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                    }
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm++) {
                        #pragma unroll
                        for (int rn = 0; rn < REGIS_N; rn++) {
                            regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                        }
                    }
                }
                
                //load the data in the register buffers to tiles
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA;
                        tileA[writeStageFlag][readRowA + i][readColA] = bufferA[loadIndex];
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB;
                        tileB[writeStageFlag][readRowB + i][readColB] = bufferB[loadIndex];
                    }
                    
                    __syncthreads();
                    writeStageFlag ^= 1;  //switch
                }
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[0][rm]) = toFloat4R(
                            tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[0][rn]) = toFloat4R(
                            tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                    }
                }
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                        C->elements[nDim * M * N + (blockM + threadIdx.y * REGIS_M + rm) * N
                                    + blockN + threadIdx.x * REGIS_N + rn] = regisC[rm][rn];
                        regisC[rm][rn] = 0;
                    }
                }
            }
        }
    }


/**
 * The first matrix is transposed before computation
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 *
 * [Unit Test Passed]
 */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetchingNT(Tensor *A, Tensor *B, Tensor *C){
        const uint32 M = A->dims.h;
        const uint32 N = B->dims.h;
        const uint32 K = A->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float* ptrA = A->elements;
        float* ptrB = B->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_K;
        const int readThreadPerRowB = BLOCK_K;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        #pragma unroll
        for(int i=0; i<BLOCK_M; i+= readRowStrideA){
            if(blockM + readRowA + i < M && readColA < K){
                tileA[0][readColA][readRowA+i] = ptrA[(blockM + readRowA + i)*K + readColA];
            }else{
                tileA[0][readColA][readRowA+i] = 0;
            }
        }
        
        #pragma unroll
        for(int i=0; i<BLOCK_N; i+= readRowStrideB){
            if(blockN + readRowB + i < N && readColB < K){
                tileB[0][readColB][readRowB+i] = ptrB[(blockN + readRowB + i)*K + readColB];
            }else{
                tileB[0][readColB][readRowB+i] = 0;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm += 4){
            toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
        }
        
        #pragma unroll
        for(int rn = 0; rn < REGIS_N; rn += 4){
            toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
        }
        
        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID+=BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                    int loadIndex = i / readRowStrideA;
                    bufferA[loadIndex] = blockM + readRowA + i < M && readColA + nextTileID < K ?
                                         ptrA[(blockM + readRowA + i) * K + readColA + nextTileID] : 0;
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                    int loadIndex = i / readRowStrideB;
                    bufferB[loadIndex] = blockN + readRowB + i < N && readColB + nextTileID < K ?
                                         ptrB[(blockN + readRowB + i) * K + readColB + nextTileID] : 0;
                }
            }
            
            int nextStageFlag = writeStageFlag ^ 1;
            
            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                            tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                            tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                    }
                }
            }
            
            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                    int loadIndex = i / readRowStrideA;
                    tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                    int loadIndex = i / readRowStrideB;
                    tileB[writeStageFlag][readColB][readRowB+i] = bufferB[loadIndex];
                }
                
                __syncthreads();
                writeStageFlag ^= 1;  //switch
            }
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(
                        tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(
                        tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }
            
            #pragma unroll
            for(int rm = 0; rm < REGIS_M; rm ++){
                #pragma unroll
                for(int rn = 0; rn < REGIS_N; rn ++){
                    regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                }
            }
        }
        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm ++){
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn ++){
                if((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                + blockN + threadIdx.x * REGIS_N + rn] = regisC[rm][rn];
                }
            }
        }
    }
    
    //[Unit Test Passed]
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetchingNTA(Tensor *A, Tensor *B, Tensor *C){
        const uint32 M = A->dims.h;
        const uint32 N = B->dims.h;
        const uint32 K = A->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float* ptrA = A->elements;
        float* ptrB = B->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_K;
        const int readThreadPerRowB = BLOCK_K;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        //for parallel processing in batch norm:
        #pragma unroll
        for(uint32 nDim = 0; nDim < A->dims.n; nDim++){
            uint32 nShiftA = nDim * M * K;
            uint32 nShiftB = nDim * K * N;
            #pragma unroll
            for(int i=0; i<BLOCK_M; i+= readRowStrideA){
                if(blockM + readRowA + i < M && readColA < K){
                    tileA[0][readColA][readRowA+i] = ptrA[nShiftA + (blockM + readRowA + i)*K + readColA];
                } else {
                    tileA[0][readColA][readRowA+i] = 0;
                }
            }
            
            #pragma unroll
            for(int i=0; i<BLOCK_N; i+= readRowStrideB){
                if(blockN + readRowB + i < N && readColB < K){
                    tileB[0][readColB][readRowB+i] = ptrB[nShiftB + (blockN + readRowB + i)*K + readColB];
                } else {
                    tileB[0][readColB][readRowB+i] = 0;
                }
            }
            __syncthreads();
            
            #pragma unroll
            for(int rm = 0; rm < REGIS_M; rm += 4){
                toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn += 4){
                toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
            }
            
            ///main loop
            int writeStageFlag = 1;
            #pragma unroll
            for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID+=BLOCK_K) {
                //prefetch
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA;
                        bufferA[loadIndex] = blockM + readRowA + i < M && readColA + nextTileID < K ?
                                             ptrA[nShiftA + (blockM + readRowA + i) * K + readColA + nextTileID] : 0;
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB;
                        bufferB[loadIndex] = blockN + readRowB + i < N && readColB + nextTileID < K ?
                                             ptrB[nShiftB + (blockN + readRowB + i) * K + readColB + nextTileID] : 0;
                    }
                }
                
                int nextStageFlag = writeStageFlag ^ 1;
                
                //compute the part that is already in the registers and load the next segment
                #pragma unroll
                for (int i = 0; i < BLOCK_K - 1; i++) {
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm += 4) {
                        toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                                tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                    }
                    
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn += 4) {
                        toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                                tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                    }
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm++) {
                        #pragma unroll
                        for (int rn = 0; rn < REGIS_N; rn++) {
                            regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                        }
                    }
                }
                
                //load the data in the register buffers to tiles
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA;
                        tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB;
                        tileB[writeStageFlag][readColB][readRowB+i] = bufferB[loadIndex];
                    }
                    
                    __syncthreads();
                    writeStageFlag ^= 1;  //switch
                }
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[0][rm]) = toFloat4R(
                            tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[0][rn]) = toFloat4R(
                            tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for(int rm = 0; rm < REGIS_M; rm ++){
                    #pragma unroll
                    for(int rn = 0; rn < REGIS_N; rn ++){
                        regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                    }
                }
            }
        }
        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm ++){
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn ++){
                if((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                + blockN + threadIdx.x * REGIS_N + rn] += regisC[rm][rn];
                }
            }
        }
    }


/**
 * The fast gemm that utilized smem and registers with data prefetching
 * @tparam BLOCK_M block size m
 * @tparam BLOCK_N block size n
 * @tparam BLOCK_K block size k
 * @tparam REGIS_M (the size of the sub matrix of C each thread compute : rows)
 * @tparam REGIS_N (the size of the sub matrix of C each thread compute : cols)
 * @param A
 * @param B
 * @param C
 *
 * [Unit Test Passed]
 */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetching4NN(Tensor *A, Tensor *B, Tensor *C) {
        
        const uint32 M = A->dims.h;
        const uint32 N = B->dims.w;
        const uint32 K = A->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_K / 4;
        const int readThreadPerRowB = BLOCK_N / 4;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA * 4;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB * 4;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        #pragma unroll
        for(uint32 nDim = 0; nDim < B->dims.n; nDim++) {
            uint32 nShift = nDim * K * N;
            float* ptrA = A->elements + blockIdx.y * BLOCK_M * K;
            float* ptrB = nShift + B->elements + blockIdx.x * BLOCK_N;
            ///prefetch the first smem and register block before starting the main loop
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA * 4;
                if (blockM + readRowA + i < M && readColA < K) {
                    toFloat4R(bufferA[loadIndex]) = toFloat4R(ptrA[(readRowA + i) * K + readColA]);
                    //transpose
                    tileA[0][readColA][readRowA + i] = bufferA[loadIndex];
                    tileA[0][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                    tileA[0][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                    tileA[0][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
                }else{
                    tileA[0][readColA][readRowA + i] = 0;
                    tileA[0][readColA + 1][readRowA + i] = 0;
                    tileA[0][readColA + 2][readRowA + i] = 0;
                    tileA[0][readColA + 3][readRowA + i] = 0;
                }
            }
            
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                if (readRowB + i < K && blockN + readColB < N) {
                    toFloat4R(tileB[0][readRowB + i][readColB]) = toFloat4R(ptrB[(readRowB + i) * N + readColB]);
                }else{
                    tileB[0][readRowB + i][readColB] = 0;
                    tileB[0][readRowB + i][readColB + 1] = 0;
                    tileB[0][readRowB + i][readColB + 2] = 0;
                    tileB[0][readRowB + i][readColB + 3] = 0;
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
            }
            
            ///main loop
            int writeStageFlag = 1;
            #pragma unroll
            for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
                //prefetch
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA * 4;
                        if (blockM + readRowA + i < M && readColA + nextTileID < K) {
                            toFloat4R(bufferA[loadIndex]) = toFloat4R(
                                    ptrA[(readRowA + i) * K + readColA + nextTileID]);
                        } else {
                            bufferA[loadIndex] = 0;
                            bufferA[loadIndex + 1] = 0;
                            bufferA[loadIndex + 2] = 0;
                            bufferA[loadIndex + 3] = 0;
                        }
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB * 4;
                        if (readRowB + i + nextTileID < K && blockN + readColB < N) {
                            toFloat4R(bufferB[loadIndex]) = toFloat4R(
                                    ptrB[(readRowB + i + nextTileID) * N + readColB]);
                        } else {
                            bufferB[loadIndex] = 0;
                            bufferB[loadIndex + 1] = 0;
                            bufferB[loadIndex + 2] = 0;
                            bufferB[loadIndex + 3] = 0;
                        }
                    }
                }
                
                int nextStageFlag = writeStageFlag ^ 1;
                
                //compute the part that is already in the registers and load the next segment
                #pragma unroll
                for (int i = 0; i < BLOCK_K - 1; i++) {
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm += 4) {
                        toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                                tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                    }
                    
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn += 4) {
                        toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                                tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                    }
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm++) {
                        #pragma unroll
                        for (int rn = 0; rn < REGIS_N; rn++) {
                            regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                        }
                    }
                }
                
                //load the data in the register buffers to tiles
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA * 4;
                        tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                        tileA[writeStageFlag][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                        tileA[writeStageFlag][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                        tileA[writeStageFlag][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideA * 4;
                        toFloat4R(tileB[writeStageFlag][readRowB + i][readColB]) = toFloat4R(bufferB[loadIndex]);
                    }
                    
                    __syncthreads();
                    writeStageFlag ^= 1;  //switch
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[0][rm]) = toFloat4R(
                            tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[0][rn]) = toFloat4R(
                            tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                    }
                }
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                        toFloat4R(C->elements[nDim * M * N + (blockM + threadIdx.y * REGIS_M + rm) * N
                                              + blockN + threadIdx.x * REGIS_N + rn]) = toFloat4R(regisC[rm][rn]);
                        regisC[rm][rn] = 0;
                        regisC[rm][rn + 1] = 0;
                        regisC[rm][rn + 2] = 0;
                        regisC[rm][rn + 3] = 0;
                    }
                }
            }
        }
    }

/**
 * The first matrix is transposed before computation
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 *
 * [Unit Test Passed]
 */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetching4TN(Tensor *A, Tensor *B, Tensor *C){
        const uint32 M = A->dims.w;
        const uint32 N = B->dims.w;
        const uint32 K = A->dims.h;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float* ptrA = A->elements;
        float* ptrB = B->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_M / 4;
        const int readThreadPerRowB = BLOCK_N / 4;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA * 4;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB * 4;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        for(uint32 nDim = 0; nDim < B->dims.n; nDim++) {
            uint32 nShift = nDim * K * N;
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                if (readRowA + i < K && blockM + readColA < M) {
                    //The mat A is not transposed since it will be transposed in smem
                    toFloat4R(tileA[0][readRowA + i][readColA]) = toFloat4R(
                            ptrA[(readRowA + i) * M + blockM + readColA]);
                }else{
                    tileA[0][readRowA + i][readColA] = 0;
                    tileA[0][readRowA + i][readColA + 1] = 0;
                    tileA[0][readRowA + i][readColA + 2] = 0;
                    tileA[0][readRowA + i][readColA + 3] = 0;
                }
            }
            
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                if (readRowB + i < K && blockN + readColB < N) {
                    toFloat4R(tileB[0][readRowB + i][readColB]) = toFloat4R(
                            ptrB[nShift + (readRowB + i) * N + blockN + readColB]);
                }else{
                    tileB[0][readRowB + i][readColB] = 0;
                    tileB[0][readRowB + i][readColB + 1] = 0;
                    tileB[0][readRowB + i][readColB + 2] = 0;
                    tileB[0][readRowB + i][readColB + 3] = 0;
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
            }
            
            
            ///main loop
            int writeStageFlag = 1;
            #pragma unroll
            for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
                //prefetch
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA * 4;
                        //here the mat A is automatially transposed while reading
                        if (readRowA + i + nextTileID < K && blockM + readColA < M) {
                            toFloat4R(bufferA[loadIndex]) = toFloat4R(
                                    ptrA[(readRowA + i + nextTileID) * M + blockM + readColA]);
                        } else {
                            bufferA[loadIndex] = 0;
                            bufferA[loadIndex + 1] = 0;
                            bufferA[loadIndex + 2] = 0;
                            bufferA[loadIndex + 3] = 0;
                        }
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB * 4;
                        if (readRowB + i + nextTileID < K && blockN + readColB < N) {
                            toFloat4R(bufferB[loadIndex]) = toFloat4R(
                                    ptrB[nShift + (readRowB + i + nextTileID) * N + blockN + readColB]);
                        } else {
                            bufferB[loadIndex] = 0;
                            bufferB[loadIndex + 1] = 0;
                            bufferB[loadIndex + 2] = 0;
                            bufferB[loadIndex + 3] = 0;
                        }
                    }
                }
                
                int nextStageFlag = writeStageFlag ^ 1;
                
                //compute the part that is already in the registers and load the next segment
                #pragma unroll
                for (int i = 0; i < BLOCK_K - 1; i++) {
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm += 4) {
                        toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                                tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                    }
                    
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn += 4) {
                        toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                                tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                    }
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm++) {
                        #pragma unroll
                        for (int rn = 0; rn < REGIS_N; rn++) {
                            regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                        }
                    }
                }
                
                //load the data in the register buffers to tiles
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA * 4;
                        toFloat4R(tileA[writeStageFlag][readRowA + i][readColA]) = toFloat4R(bufferA[loadIndex]);
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB * 4;
                        toFloat4R(tileB[writeStageFlag][readRowB + i][readColB]) = toFloat4R(bufferB[loadIndex]);
                    }
                    
                    __syncthreads();
                    writeStageFlag ^= 1;  //switch
                }
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[0][rm]) = toFloat4R(
                            tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[0][rn]) = toFloat4R(
                            tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                    }
                }
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                        toFloat4R(C->elements[nDim * M * N + (blockM + threadIdx.y * REGIS_M + rm) * N +
                                              blockN + threadIdx.x * REGIS_N + rn])
                                = toFloat4R(regisC[rm][rn]);
                        regisC[rm][rn] = 0;
                        regisC[rm][rn + 1] = 0;
                        regisC[rm][rn + 2] = 0;
                        regisC[rm][rn + 3] = 0;
                    }
                }
            }
        }
    }
    
    //[Unit Test Passed]
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetching4NT(Tensor* A, Tensor* B, Tensor* C) {
        const uint32 M = A->dims.h;
        const uint32 N = B->dims.h;
        const uint32 K = A->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float *ptrA = A->elements;
        float *ptrB = B->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_K / 4;
        const int readThreadPerRowB = BLOCK_K / 4;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA * 4;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB * 4;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
            int loadIndex = i / readRowStrideA * 4;
            if (blockM + readRowA + i < M && readColA < K) {
                toFloat4R(bufferA[loadIndex]) = toFloat4R(ptrA[(blockM + readRowA + i) * K + readColA]);
                //transpose
                tileA[0][readColA][readRowA + i] = bufferA[loadIndex];
                tileA[0][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                tileA[0][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                tileA[0][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
            }else{
                tileA[0][readColA][readRowA + i] = 0;
                tileA[0][readColA + 1][readRowA + i] = 0;
                tileA[0][readColA + 2][readRowA + i] = 0;
                tileA[0][readColA + 3][readRowA + i] = 0;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
            int loadIndex = i / readRowStrideB * 4;
            if (blockN + readRowB + i < N && readColB < K) {
                toFloat4R(bufferB[loadIndex]) = toFloat4R(ptrB[(blockN + readRowB + i) * K + readColB]);
                
                tileB[0][readColB][readRowB + i] = bufferB[loadIndex];
                tileB[0][readColB + 1][readRowB + i] = bufferB[loadIndex + 1];
                tileB[0][readColB + 2][readRowB + i] = bufferB[loadIndex + 2];
                tileB[0][readColB + 3][readRowB + i] = bufferB[loadIndex + 3];
            }else{
                tileB[0][readColB][readRowB + i] = 0;
                tileB[0][readColB + 1][readRowB + i] = 0;
                tileB[0][readColB + 2][readRowB + i] = 0;
                tileB[0][readColB + 3][readRowB + i] = 0;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
        }
        
        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
        }
        
        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                    int loadIndex = i / readRowStrideA * 4;
                    if (blockM + readRowA + i < M && readColA + nextTileID < K) {
                        toFloat4R(bufferA[loadIndex]) = toFloat4R(
                                ptrA[(readRowA + i) * K + readColA + nextTileID]);
                    } else {
                        bufferA[loadIndex] = 0;
                        bufferA[loadIndex + 1] = 0;
                        bufferA[loadIndex + 2] = 0;
                        bufferA[loadIndex + 3] = 0;
                    }
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                    int loadIndex = i / readRowStrideB * 4;
                    if (blockN + readRowB + i < N && readColB + nextTileID < K) {
                        toFloat4R(bufferB[loadIndex]) =
                                toFloat4R(ptrB[(blockN + readRowB + i) * K + readColB + nextTileID]);
                    } else {
                        bufferB[loadIndex] = 0;
                        bufferB[loadIndex + 1] = 0;
                        bufferB[loadIndex + 2] = 0;
                        bufferB[loadIndex + 3] = 0;
                    }
                }
            }
            
            int nextStageFlag = writeStageFlag ^ 1;
            
            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                            tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                            tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                    }
                }
            }
            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                    int loadIndex = i / readRowStrideA * 4;
                    tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                    tileA[writeStageFlag][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                    tileA[writeStageFlag][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                    tileA[writeStageFlag][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                    int loadIndex = i / readRowStrideA * 4;
                    tileB[writeStageFlag][readColB][readRowB + i] = bufferB[loadIndex];
                    tileB[writeStageFlag][readColB + 1][readRowB + i] = bufferB[loadIndex + 1];
                    tileB[writeStageFlag][readColB + 2][readRowB + i] = bufferB[loadIndex + 2];
                    tileB[writeStageFlag][readColB + 3][readRowB + i] = bufferB[loadIndex + 3];
                }
                
                __syncthreads();
                writeStageFlag ^= 1;
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(
                        tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(
                        tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }
            
            #pragma unroll
            for(int rm = 0; rm < REGIS_M; rm ++){
                #pragma unroll
                for(int rn = 0; rn < REGIS_N; rn ++){
                    regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                }
            }
        }
        
        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm ++){
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn ++){
                if((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                + blockN + threadIdx.x * REGIS_N + rn] = regisC[rm][rn];
                }
            }
        }
    }
    
    //[Unit Test Passed]
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmPrefetching4NTA (Tensor* A, Tensor* B, Tensor* C) {
        const uint32 M = A->dims.h;
        const uint32 N = B->dims.h;
        const uint32 K = A->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];
        
        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float *ptrA = A->elements;
        float *ptrB = B->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowA = BLOCK_K / 4;
        const int readThreadPerRowB = BLOCK_K / 4;
        
        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA * 4;
        
        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB * 4;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;
        
        //for parallel processing in batch norm:
        #pragma unroll
        for(uint32 nDim = 0; nDim < A->dims.n; nDim++){
            uint32 nShiftA = nDim * M * K;
            uint32 nShiftB = nDim * N * K;
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA * 4;
                if (blockM + readRowA + i < M && readColA < K) {
                    toFloat4R(bufferA[loadIndex]) = toFloat4R(ptrA[nShiftA + (blockM + readRowA + i) * K + readColA]);
                    //transpose
                    tileA[0][readColA][readRowA + i] = bufferA[loadIndex];
                    tileA[0][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                    tileA[0][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                    tileA[0][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
                }else{
                    tileA[0][readColA][readRowA + i] = 0;
                    tileA[0][readColA + 1][readRowA + i] = 0;
                    tileA[0][readColA + 2][readRowA + i] = 0;
                    tileA[0][readColA + 3][readRowA + i] = 0;
                }
            }
            
            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB * 4;
                if (blockN + readRowB + i < N && readColB < K) {
                    toFloat4R(bufferB[loadIndex]) = toFloat4R(ptrB[nShiftB + (blockN + readRowB + i) * K + readColB]);
                    
                    tileB[0][readColB][readRowB + i] = bufferB[loadIndex];
                    tileB[0][readColB + 1][readRowB + i] = bufferB[loadIndex + 1];
                    tileB[0][readColB + 2][readRowB + i] = bufferB[loadIndex + 2];
                    tileB[0][readColB + 3][readRowB + i] = bufferB[loadIndex + 3];
                }else{
                    tileB[0][readColB][readRowB + i] = 0;
                    tileB[0][readColB + 1][readRowB + i] = 0;
                    tileB[0][readColB + 2][readRowB + i] = 0;
                    tileB[0][readColB + 3][readRowB + i] = 0;
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
            }
            
            ///main loop
            int writeStageFlag = 1;
            #pragma unroll
            for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
                //prefetch
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA * 4;
                        if (blockM + readRowA + i < M && readColA + nextTileID < K) {
                            toFloat4R(bufferA[loadIndex]) = toFloat4R(
                                    ptrA[nShiftA +(readRowA + i) * K + readColA + nextTileID]);
                        } else {
                            bufferA[loadIndex] = 0;
                            bufferA[loadIndex + 1] = 0;
                            bufferA[loadIndex + 2] = 0;
                            bufferA[loadIndex + 3] = 0;
                        }
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideB * 4;
                        if (blockN + readRowB + i < N && readColB + nextTileID < K) {
                            toFloat4R(bufferB[loadIndex]) =
                                    toFloat4R(ptrB[nShiftB + (blockN + readRowB + i) * K + readColB + nextTileID]);
                        } else {
                            bufferB[loadIndex] = 0;
                            bufferB[loadIndex + 1] = 0;
                            bufferB[loadIndex + 2] = 0;
                            bufferB[loadIndex + 3] = 0;
                        }
                    }
                }
                
                int nextStageFlag = writeStageFlag ^ 1;
                
                //compute the part that is already in the registers and load the next segment
                #pragma unroll
                for (int i = 0; i < BLOCK_K - 1; i++) {
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm += 4) {
                        toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                                tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                    }
                    
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn += 4) {
                        toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                                tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                    }
                    
                    #pragma unroll
                    for (int rm = 0; rm < REGIS_M; rm++) {
                        #pragma unroll
                        for (int rn = 0; rn < REGIS_N; rn++) {
                            regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                        }
                    }
                }
                //load the data in the register buffers to tiles
                if (nextTileID < K) {
                    #pragma unroll
                    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                        int loadIndex = i / readRowStrideA * 4;
                        tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                        tileA[writeStageFlag][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                        tileA[writeStageFlag][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                        tileA[writeStageFlag][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                        int loadIndex = i / readRowStrideA * 4;
                        tileB[writeStageFlag][readColB][readRowB + i] = bufferB[loadIndex];
                        tileB[writeStageFlag][readColB + 1][readRowB + i] = bufferB[loadIndex + 1];
                        tileB[writeStageFlag][readColB + 2][readRowB + i] = bufferB[loadIndex + 2];
                        tileB[writeStageFlag][readColB + 3][readRowB + i] = bufferB[loadIndex + 3];
                    }
                    
                    __syncthreads();
                    writeStageFlag ^= 1;
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[0][rm]) = toFloat4R(
                            tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[0][rn]) = toFloat4R(
                            tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                    }
                }
            }
        }
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm++) {
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn ++) {
                if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    float cSrc = C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N + blockN + threadIdx.x * REGIS_N + rn];
                    C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N + blockN + threadIdx.x * REGIS_N + rn]
                            = regisC[rm][rn]+ cSrc;
                }
            }
        }
    }
    
    Tensor* sgemm(Tensor *A, Tensor *B, Tensor *C) {
        assert(A->dims.w == B->dims.h);
        assert(A->dims.h == C->dims.h);
        assert(B->dims.w == C->dims.w);
        dim3 grid = dim3((C->dims.w + BN - 1) / BN, (C->dims.h + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        if(A->dims.w%4==0 && B->dims.w%4==0){
            gemmPrefetching4NN < BM, BN, BK, RM, RN ><<<grid, block>>>(A, B, C);
        } else {
            gemmPrefetchingNN < BM, BN, BK, RM, RN ><<<grid, block>>>(A, B, C);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return C;
    }
    
    //gemm with the first matrix automatically transposed
    Tensor* sgemmTN(Tensor *A, Tensor *B, Tensor *C) {
        assert(A->dims.h == B->dims.h);
        assert(A->dims.w == C->dims.h && B->dims.w == C->dims.w);
        
        dim3 grid = dim3((C->dims.w + BN - 1) / BN, (C->dims.h + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        if(A->dims.w%4==0 && B->dims.w%4==0){
            gemmPrefetching4TN <BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C);
        } else {
            gemmPrefetchingTN<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return C;
    }
    
    //gemm with the second matrix transposed
    Tensor* sgemmNT(Tensor *A, Tensor *B, Tensor *C) {
        assert(A->dims.w == B->dims.w);
        assert(A->dims.h == C->dims.h && B->dims.h == C->dims.w);
        
        dim3 grid = dim3((C->dims.w + BN - 1) / BN, (C->dims.h + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        if(A->dims.w%4 == 0 && B->dims.w%4 == 0){
            gemmPrefetching4NT<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
        }else
            gemmPrefetchingNT<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return C;
    }
    
    Tensor* sgemmNTA(Tensor *A, Tensor *B, Tensor *C) {
        assert(A->dims.w == B->dims.w);
        assert(A->dims.h == C->dims.h && B->dims.h == C->dims.w);
        dim3 grid = dim3((C->dims.w + BN - 1) / BN, (C->dims.h + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        if(A->dims.w%4 == 0 && B->dims.w%4 == 0){
            gemmPrefetching4NTA<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
        }else
            gemmPrefetchingNTA<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return C;
    }
} // seblas