//
// Created by Dylan on 6/8/2022.
//

#include "cuLinear.cuh"
#include "../assist/Inspections.cuh"

#define BM 128
#define BN 128
#define BK 8
#define RM 8
#define RN 8

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

namespace seblas {

/**
 * The linear operation
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
    __global__ void linearD(Tensor *X, Tensor *W, Tensor *B, Tensor *Y) {
        const uint32 M = W->dims.h;
        const uint32 N = X->dims.w * X->dims.n;
        const uint32 K = W->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileW[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileX[2][BLOCK_K][BLOCK_N];
        
        float regisW[2][REGIS_M];
        float regisX[2][REGIS_N];
        float regisY[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferW[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferX[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowW = BLOCK_K;
        const int readThreadPerRowX = BLOCK_K;
        
        //the location each thread should be reading relative to smem
        const int readRowW = tid / readThreadPerRowW;
        const int readColW = tid % readThreadPerRowW;
        
        const int readRowX = tid / readThreadPerRowX;
        const int readColX = tid % readThreadPerRowX;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideW = threadCount / readThreadPerRowW;
        const int readRowStrideX = threadCount / readThreadPerRowX;
        
        float *ptrW = W->elements;
        float *ptrX = X->elements;
        
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += readRowStrideW) {
            if (blockM + readRowW + i < M && readColW < K) {
                tileW[0][readColW][readRowW + i] = ptrW[(blockM + readRowW + i) * K + readColW];
            }else{
                tileW[0][readColW][readRowW + i] = 0;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i += readRowStrideX) {
            if (blockN + readRowX + i < N && readColX < K) {
                tileX[0][readColX][readRowX + i] = ptrX[(blockN + readRowX + i)*K + readColX];
            }else{
                tileX[0][readColX][readRowX + i] = 0;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisW[0][rm]) = toFloat4R(tileW[0][0][REGIS_M * threadIdx.y + rm]);
        }
        
        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisX[0][rn]) = toFloat4R(tileX[0][0][REGIS_N * threadIdx.x + rn]);
        }
        
        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW;
                    bufferW[loadIndex] = blockM + readRowW + i < M && readColW + nextTileID < K ?
                                         ptrW[(blockM + readRowW + i) * K + readColW + nextTileID] : 0;
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideX) {
                    int loadIndex = i / readRowStrideX;
                    bufferX[loadIndex] =  blockN + readRowX + i < N && readColX + nextTileID < K ?
                                         ptrX[(blockN + readRowX + i) * K + readColX + nextTileID] : 0;
                }
            }
            
            int nextStageFlag = writeStageFlag ^ 1;
            
            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisW[(i + 1) % 2][rm]) = toFloat4R(
                            tileW[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisX[(i + 1) % 2][rn]) = toFloat4R(
                            tileX[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisY[rm][rn] += regisW[i % 2][rm] * regisX[i % 2][rn];
                    }
                }
            }
            
            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW;
                    tileW[writeStageFlag][readColW][readRowW + i] = bufferW[loadIndex];
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideX) {
                    int loadIndex = i / readRowStrideX;
                    tileX[writeStageFlag][readColX][readRowX + i] = bufferX[loadIndex];
                }
                
                __syncthreads();
                writeStageFlag ^= 1;  //switch
            }
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisW[0][rm]) = toFloat4R(
                        tileW[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisX[0][rn]) = toFloat4R(
                        tileX[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    regisY[rm][rn] += regisW[1][rm] * regisX[1][rn];
                }
            }
            
            
            uint32 nShift = Y->dims.size / Y->dims.n;
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                float bVal = B->elements[blockM + threadIdx.y * REGIS_M + rm];
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                        Y->elements[blockM + threadIdx.y * REGIS_M + rm
                                    + (blockN + threadIdx.x * REGIS_N + rn) * nShift] = regisY[rm][rn] + bVal;
                    }
                }
            }
        }
    }


/**
 * Prefetching with LDS.128 memread acceleration
 * @tparam BLOCK_M block size m
 * @tparam BLOCK_N block size n
 * @tparam BLOCK_K block size k
 * @tparam REGIS_M (the size of the sub matrix of Y each thread compute : rows)
 * @tparam REGIS_N (the size of the sub matrix of Y each thread compute : cols)
 * @param W
 * @param X
 * @param Y
 *
 * [Unit Test Passed]
 */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void linear4D( Tensor *X, Tensor *W, Tensor *B, Tensor *Y) {
        
        //here, we view A as A n x h matrix, and we transpose it while reading
        const uint32 M = W->dims.h;
        const uint32 N = X->dims.n;
        const uint32 K = W->dims.w;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileW[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileX[2][BLOCK_K][BLOCK_N];
        
        float regisW[2][REGIS_M];
        float regisX[2][REGIS_N];
        float regisY[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferW[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferX[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowW = BLOCK_K / 4;
        const int readThreadPerRowX = BLOCK_K / 4;
        
        //the location each thread should be reading relative to smem
        const int readRowW = tid / readThreadPerRowW;
        const int readColW = tid % readThreadPerRowW * 4;
        
        const int readRowX = tid / readThreadPerRowX;
        const int readColX = tid % readThreadPerRowX * 4;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideW = threadCount / readThreadPerRowW;
        const int readRowStrideX = threadCount / readThreadPerRowX;
        
        float* ptrW = W->elements + blockIdx.y * BLOCK_M * K;
        float* ptrX = X->elements;
        ///prefetch the first smem and register block before starting the main loop
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += readRowStrideW) {
            int loadIndex = i / readRowStrideW * 4;
            if (blockM + readRowW + i < M && readColW < K) {
                toFloat4R(bufferW[loadIndex]) = toFloat4R(ptrW[(readRowW + i) * K + readColW]);
                //transpose
                tileW[0][readColW][readRowW + i] = bufferW[loadIndex];
                tileW[0][readColW + 1][readRowW + i] = bufferW[loadIndex + 1];
                tileW[0][readColW + 2][readRowW + i] = bufferW[loadIndex + 2];
                tileW[0][readColW + 3][readRowW + i] = bufferW[loadIndex + 3];
            }else{
                tileW[0][readColW][readRowW + i] = 0;
                tileW[0][readColW + 1][readRowW + i] = 0;
                tileW[0][readColW + 2][readRowW + i] = 0;
                tileW[0][readColW + 3][readRowW + i] = 0;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += readRowStrideX) {
            int loadIndex = i / readRowStrideX * 4;
            if (blockN + readRowX + i < N && readColX < K) {
                toFloat4R(bufferX[loadIndex]) = toFloat4R(ptrX[(blockN + readRowX + i) * K + readColX]);
                
                tileX[0][readColX][readRowX + i] = bufferX[loadIndex];
                tileX[0][readColX + 1][readRowX + i] = bufferX[loadIndex + 1];
                tileX[0][readColX + 2][readRowX + i] = bufferX[loadIndex + 2];
                tileX[0][readColX + 3][readRowX + i] = bufferX[loadIndex + 3];
                
            }else{
                tileX[0][readColX][readRowX + i] = 0;
                tileX[0][readColX + 1][readRowX + i] = 0;
                tileX[0][readColX + 2][readRowX + i] = 0;
                tileX[0][readColX + 3][readRowX + i] = 0;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisW[0][rm]) = toFloat4R(tileW[0][0][REGIS_M * threadIdx.y + rm]);
        }
        
        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisX[0][rn]) = toFloat4R(tileX[0][0][REGIS_N * threadIdx.x + rn]);
        }
        
        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW * 4;
                    if (blockM + readRowW + i < M && readColW + nextTileID < K) {
                        toFloat4R(bufferW[loadIndex]) = toFloat4R(
                                ptrW[(readRowW + i) * K + readColW + nextTileID]);
                    } else {
                        bufferW[loadIndex] = 0;
                        bufferW[loadIndex + 1] = 0;
                        bufferW[loadIndex + 2] = 0;
                        bufferW[loadIndex + 3] = 0;
                    }
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideX) {
                    int loadIndex = i / readRowStrideX * 4;
                    if (blockN + readRowX + i < N && readColX + nextTileID < K) {
                        toFloat4R(bufferX[loadIndex]) =
                                toFloat4R(ptrX[(blockN + readRowX + i) * K + readColX + nextTileID]);
                    } else {
                        bufferX[loadIndex] = 0;
                        bufferX[loadIndex + 1] = 0;
                        bufferX[loadIndex + 2] = 0;
                        bufferX[loadIndex + 3] = 0;
                    }
                }
            }
            
            int nextStageFlag = writeStageFlag ^ 1;
            
            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisW[(i + 1) % 2][rm]) = toFloat4R(
                            tileW[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisX[(i + 1) % 2][rn]) = toFloat4R(
                            tileX[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisY[rm][rn] += regisW[i % 2][rm] * regisX[i % 2][rn];
                    }
                }
            }
            
            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW * 4;
                    tileW[writeStageFlag][readColW][readRowW + i] = bufferW[loadIndex];
                    tileW[writeStageFlag][readColW + 1][readRowW + i] = bufferW[loadIndex + 1];
                    tileW[writeStageFlag][readColW + 2][readRowW + i] = bufferW[loadIndex + 2];
                    tileW[writeStageFlag][readColW + 3][readRowW + i] = bufferW[loadIndex + 3];
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideX) {
                    int loadIndex = i / readRowStrideW * 4;
                    
                    tileX[writeStageFlag][readColX][readRowX + i] = bufferX[loadIndex];
                    tileX[writeStageFlag][readColX + 1][readRowX + i] = bufferX[loadIndex + 1];
                    tileX[writeStageFlag][readColX + 2][readRowX + i] = bufferX[loadIndex + 2];
                    tileX[writeStageFlag][readColX + 3][readRowX + i] = bufferX[loadIndex + 3];
                }
                
                __syncthreads();
                writeStageFlag ^= 1;  //switch
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisW[0][rm]) = toFloat4R(
                        tileW[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisX[0][rn]) = toFloat4R(
                        tileX[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    regisY[rm][rn] += regisW[1][rm] * regisX[1][rn];
                }
            }
        }
        
        uint32 nShift = Y->dims.size / Y->dims.n;
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm++) {
            float bVal = B->elements[blockM + threadIdx.y * REGIS_M + rm];
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn++) {
                if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    Y->elements[blockM + threadIdx.y * REGIS_M + rm
                                + (blockN + threadIdx.x * REGIS_N + rn) * nShift] = regisY[rm][rn] + bVal;
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
  * @param W
  * @param dY
  * @param dX
   *
   * [Unit Test Passed]
  */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void linearXGradD(Tensor *W, Tensor *dY, Tensor *dX){
        const uint32 M = W->dims.w;
        const uint32 N = dY->dims.w * dY->dims.n;
        const uint32 K = W->dims.h;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileW[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileDY[2][BLOCK_K][BLOCK_N];
        
        float regisW[2][REGIS_M];
        float regisDY[2][REGIS_N];
        float regisDX[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferW[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferDY[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float* ptrW = W->elements;
        float* ptrDY = dY->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowW = BLOCK_M;
        const int readThreadPerRowDY = BLOCK_K;
        
        //the location each thread should be reading relative to smem
        const int readRowW = tid / readThreadPerRowW;
        const int readColW = tid % readThreadPerRowW;
        
        const int readRowDY = tid / readThreadPerRowDY;
        const int readColDY = tid % readThreadPerRowDY;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideW = threadCount / readThreadPerRowW;
        const int readRowStrideDY = threadCount / readThreadPerRowDY;
        
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += readRowStrideW) {
            if (readRowW + i < K && blockM + readColW < M) {
                //The mat W is not transposed since it will be transposed in smem
                tileW[0][readRowW + i][readColW] = ptrW[(readRowW + i) * M + blockM + readColW];
            } else {
                tileW[0][readRowW + i][readColW] = 0;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i += readRowStrideDY) {
            if (blockN + readRowDY + i < N && readColDY < K) {
                tileDY[0][readColDY][readRowDY + i] = ptrDY[(blockN + readRowDY + i) * K + readColDY];
            } else {
                tileDY[0][readColDY][readRowDY + i] = 0;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisW[0][rm]) = toFloat4R(tileW[0][0][REGIS_M * threadIdx.y + rm]);
        }
        
        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisDY[0][rn]) = toFloat4R(tileDY[0][0][REGIS_N * threadIdx.x + rn]);
        }
        
        
        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW;
                    //here the mat W is automatially transposed while reading
                    bufferW[loadIndex] = readRowW + i + nextTileID < K && blockM + readColW < M ?
                                         ptrW[(readRowW + i + nextTileID) * M + blockM + readColW] : 0;
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideDY) {
                    int loadIndex = i / readRowStrideDY;
                    bufferDY[loadIndex] = blockN + readRowDY + i < N && readColDY + nextTileID < K ?
                                          ptrDY[(readRowDY + i + blockN) * K + nextTileID + readColDY] : 0;
                }
            }
            
            int nextStageFlag = writeStageFlag ^ 1;
            
            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisW[(i + 1) % 2][rm]) = toFloat4R(
                            tileW[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisDY[(i + 1) % 2][rn]) = toFloat4R(
                            tileDY[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisDX[rm][rn] += regisW[i % 2][rm] * regisDY[i % 2][rn];
                    }
                }
            }
            
            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW;
                    tileW[writeStageFlag][readRowW + i][readColW] = bufferW[loadIndex];
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_N; i += readRowStrideDY) {
                    int loadIndex = i / readRowStrideDY;
                    tileDY[writeStageFlag][readColDY][readRowDY + i] = bufferDY[loadIndex];
                }
                
                __syncthreads();
                writeStageFlag ^= 1;  //switch
            }
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisW[0][rm]) = toFloat4R(
                        tileW[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisDY[0][rn]) = toFloat4R(
                        tileDY[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    regisDX[rm][rn] += regisW[1][rm] * regisDY[1][rn];
                }
            }
    
            uint32 nShift = dX->dims.size / dX->dims.n;
    
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                        dX->elements[blockM + threadIdx.y * REGIS_M + rm
                                    + (blockN + threadIdx.x * REGIS_N + rn) * nShift] += regisDX[rm][rn];
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
 * @param W
 * @param dY
 * @param dX
 *
 * [Unit Test Passed]
 */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void linearXGrad4D(Tensor *W, Tensor *dY, Tensor *dX){
        const uint32 M = W->dims.w;
        const uint32 N = dY->dims.w * dY->dims.n;
        const uint32 K = W->dims.h;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileW[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileDY[2][BLOCK_K][BLOCK_N];
        
        float regisW[2][REGIS_M];
        float regisDY[2][REGIS_N];
        float regisDX[REGIS_M][REGIS_N] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferW[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferDY[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float* ptrW = W->elements;
        float* ptrDY = dY->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowW = BLOCK_M / 4;
        const int readThreadPerRowDY = BLOCK_K / 4;
        
        //the location each thread should be reading relative to smem
        const int readRowW = tid / readThreadPerRowW;
        const int readColW = tid % readThreadPerRowW * 4;
        
        const int readRowDY = tid / readThreadPerRowDY;
        const int readColDY = tid % readThreadPerRowDY * 4;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideW = threadCount / readThreadPerRowW;
        const int readRowStrideDY = threadCount / readThreadPerRowDY;
    
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += readRowStrideW) {
            if (readRowW + i < K && blockM + readColW < M) {
                //The mat W is not transposed since it will be transposed in smem
                toFloat4R(tileW[0][readRowW + i][readColW]) = toFloat4R(
                        ptrW[(readRowW + i) * M + blockM + readColW]);
            }else{
                tileW[0][readRowW + i][readColW] = 0;
                tileW[0][readRowW + i][readColW + 1] = 0;
                tileW[0][readRowW + i][readColW + 2] = 0;
                tileW[0][readRowW + i][readColW + 3] = 0;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i += readRowStrideDY) {
            int loadIndex = i / readRowStrideDY * 4;
            
            if (blockN + readRowDY + i < N && readColDY < K) {
                toFloat4R(bufferDY[loadIndex]) = toFloat4R(ptrDY[(blockN + readRowDY + i) * K + readColDY]);
        
                tileDY[0][readColDY][readRowDY + i] = bufferDY[loadIndex];
                tileDY[0][readColDY + 1][readRowDY + i] = bufferDY[loadIndex + 1];
                tileDY[0][readColDY + 2][readRowDY + i] = bufferDY[loadIndex + 2];
                tileDY[0][readColDY + 3][readRowDY + i] = bufferDY[loadIndex + 3];
            }else{
                tileDY[0][readColDY][readRowDY + i] = 0;
                tileDY[0][readColDY + 1][readRowDY + i] = 0;
                tileDY[0][readColDY + 2][readRowDY + i] = 0;
                tileDY[0][readColDY + 3][readRowDY + i] = 0;
            }
            
        }
        __syncthreads();
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisW[0][rm]) = toFloat4R(tileW[0][0][REGIS_M * threadIdx.y + rm]);
        }
        
        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisDY[0][rn]) = toFloat4R(tileDY[0][0][REGIS_N * threadIdx.x + rn]);
        }
        
        
        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW * 4;
                    if (readRowW + i + nextTileID < K && blockM + readColW < M) {
                        toFloat4R(bufferW[loadIndex]) = toFloat4R(
                                ptrW[(readRowW + i + nextTileID) * M + blockM + readColW]);
                    } else {
                        bufferW[loadIndex] = 0;
                        bufferW[loadIndex + 1] = 0;
                        bufferW[loadIndex + 2] = 0;
                        bufferW[loadIndex + 3] = 0;
                    }
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideDY) {
                    int loadIndex = i / readRowStrideDY * 4;
                    
                    if (blockN + readRowDY + i < N && readColDY + nextTileID < K) {
                        toFloat4R(bufferDY[loadIndex]) =
                                toFloat4R(ptrDY[(blockN + readRowDY + i) * K + readColDY + nextTileID]);
                    } else {
                        bufferDY[loadIndex] = 0;
                        bufferDY[loadIndex + 1] = 0;
                        bufferDY[loadIndex + 2] = 0;
                        bufferDY[loadIndex + 3] = 0;
                    }
                }
            }
            
            int nextStageFlag = writeStageFlag ^ 1;
            
            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisW[(i + 1) % 2][rm]) = toFloat4R(
                            tileW[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisDY[(i + 1) % 2][rn]) = toFloat4R(
                            tileDY[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisDX[rm][rn] += regisW[i % 2][rm] * regisDY[i % 2][rn];
                    }
                }
            }
            
            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideW) {
                    int loadIndex = i / readRowStrideW * 4;
                    toFloat4R(tileW[writeStageFlag][readRowW + i][readColW]) = toFloat4R(bufferW[loadIndex]);
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideDY) {
                    int loadIndex = i / readRowStrideDY * 4;
                    tileDY[writeStageFlag][readColDY][readRowDY + i] = bufferDY[loadIndex];
                    tileDY[writeStageFlag][readColDY + 1][readRowDY + i] = bufferDY[loadIndex + 1];
                    tileDY[writeStageFlag][readColDY + 2][readRowDY + i] = bufferDY[loadIndex + 2];
                    tileDY[writeStageFlag][readColDY + 3][readRowDY + i] = bufferDY[loadIndex + 3];
                }
                
                __syncthreads();
                writeStageFlag ^= 1;  //switch
            }
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisW[0][rm]) = toFloat4R(
                        tileW[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisDY[0][rn]) = toFloat4R(
                        tileDY[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    regisDX[rm][rn] += regisW[1][rm] * regisDY[1][rn];
                }
            }
    
            uint32 nShift = dX->dims.size / dX->dims.n;
    
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                        dX->elements[blockM + threadIdx.y * REGIS_M + rm
                                     + (blockN + threadIdx.x * REGIS_N + rn) * nShift] += regisDX[rm][rn];
                    }
                }
            }
        }
    }
    
    
    /**
    * Linear Parameter Gradients
    * @tparam BLOCK_M
    * @tparam BLOCK_N
    * @tparam BLOCK_K
    * @tparam REGIS_M
    * @tparam REGIS_N
    * @param dY
    * @param X
    * @param dW
     *
     * [Unit Test Passed]
    */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void linearParamGradsD(Tensor *dY, Tensor *X, Tensor *dW, Tensor *dB){
        const uint32 M = dY->dims.h;  //dY : n * h
        const uint32 N = X->dims.h;  //A : n * h
        const uint32 K = dY->dims.n;
        
        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileDY[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileX[2][BLOCK_K][BLOCK_N];
        
        float regisDY[2][REGIS_M];
        float regisX[2][REGIS_N];
        float regisDW[REGIS_M][REGIS_N] = {0};
        float regisDB[REGIS_M] = {0};
        
        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;
        
        ///register for buffering elements during transporting global to shared mem
        float bufferDY[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferX[BLOCK_N * BLOCK_K / threadCount] = {0};
        
        ///prepare configs for reading global
        float* ptrDY = dY->elements;
        float* ptrX = X->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;
        
        const int readThreadPerRowDY = BLOCK_M;
        const int readThreadPerRowX = BLOCK_N;
        
        //the location each thread should be reading relative to smem
        const int readRowDY = tid / readThreadPerRowDY;
        const int readColDY = tid % readThreadPerRowDY;
        
        const int readRowX = tid / readThreadPerRowX;
        const int readColX = tid % readThreadPerRowX;
        
        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideDY = threadCount / readThreadPerRowDY;
        const int readRowStrideX = threadCount / readThreadPerRowX;
        
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += readRowStrideDY) {
            if (readRowDY + i < K && blockM + readColDY < M) {
                //The mat dY is not transposed since it will be transposed in smem
                tileDY[0][readRowDY + i][readColDY] = ptrDY[(readRowDY + i) * M + blockM + readColDY];
            } else {
                tileDY[0][readRowDY + i][readColDY] = 0;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += readRowStrideX) {
            if (readRowX + i < K && blockN + readColX < N) {
                tileX[0][readRowX + i][readColX] = ptrX[(readRowX + i) * N + blockN + readColX];
            } else {
                tileX[0][readRowX + i][readColX] = 0;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisDY[0][rm]) = toFloat4R(tileDY[0][0][REGIS_M * threadIdx.y + rm]);
        }
        
        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisX[0][rn]) = toFloat4R(tileX[0][0][REGIS_N * threadIdx.x + rn]);
        }
        
        
        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID += BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideDY) {
                    int loadIndex = i / readRowStrideDY;
                    //here the mat dY is automatially transposed while reading
                    bufferDY[loadIndex] = readRowDY + i + nextTileID < K && blockM + readColDY < M ?
                                          ptrDY[(readRowDY + i + nextTileID) * M + blockM + readColDY] : 0;
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideX) {
                    int loadIndex = i / readRowStrideX;
                    bufferX[loadIndex] = readRowX + i + nextTileID < K && blockN + readColX < N ?
                                         ptrX[(readRowX + i + nextTileID) * N + blockN + readColX] : 0;
                }
            }
            
            int nextStageFlag = writeStageFlag ^ 1;
            
            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisDY[(i + 1) % 2][rm]) = toFloat4R(
                            tileDY[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }
                
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisX[(i + 1) % 2][rn]) = toFloat4R(
                            tileX[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }
                
                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    regisDB[rm] += regisDY[i % 2][rm];
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisDW[rm][rn] += regisDY[i % 2][rm] * regisX[i % 2][rn];
                    }
                }
            }
            
            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideDY) {
                    int loadIndex = i / readRowStrideDY;
                    tileDY[writeStageFlag][readRowDY + i][readColDY] = bufferDY[loadIndex];
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideX) {
                    int loadIndex = i / readRowStrideX;
                    tileX[writeStageFlag][readRowX + i][readColX] = bufferX[loadIndex];
                }
                
                __syncthreads();
                writeStageFlag ^= 1;  //switch
            }
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisDY[0][rm]) = toFloat4R(
                        tileDY[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }
            
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisX[0][rn]) = toFloat4R(
                        tileX[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }
            
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                regisDB[rm] += regisDY[1][rm];
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    regisDW[rm][rn] += regisDY[1][rm] * regisX[1][rn];
                }
            }
        }
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm++) {
            if (blockM + threadIdx.y * REGIS_M + rm < M) {
                dB->elements[blockM + threadIdx.y * REGIS_M + rm] += regisDB[rm];
            }
            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn++) {
                if ((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    dW->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                 + blockN + threadIdx.x * REGIS_N + rn] += regisDW[rm][rn];
                }
            }
        }
    }
    
    //[Unit Test Passed]
    Tensor* linear(Tensor* X, Tensor* W, Tensor* B, Tensor* Y) {
        uint32 M = Y->dims.h;
        uint32 N = Y->dims.n * Y->dims.w;
    
        assert(W->dims.w == X->dims.h);
        assert(B->dims.h == M);
        assert(Y->dims.w == 1 && B->dims.w == 1 && X->dims.w==1);
        assert(X->dims.n == Y->dims.n);
    
        dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        if(W->dims.w % 4 == 0 && N % 4 == 0){
            linear4D<BM, BN, BK, RM, RN><<<grid, block>>>(X,W,B,Y);
        }else {
            linearD<BM, BN, BK, RM, RN><<<grid, block>>>(X, W, B, Y);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return Y;
    }
    
    //[Unit Test Passed]
    Tensor* linearXGrad(Tensor* dY, Tensor* W, Tensor* dX) {
        uint32 M = dX->dims.h;
        uint32 N = dX->dims.n * dX->dims.w;
        
        assert(W->dims.h == dY->dims.h);
        assert(dX->dims.n == dY->dims.n);
        assert(dX->dims.w == 1 && dY->dims.w == 1);
        assert(W->dims.w == dX->dims.h);
    
        dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
    
        if(W->dims.w % 4 == 0 && W->dims.h %4 == 0 && N % 4 == 0) {
            linearXGrad4D<BM, BN, BK, RM, RN><<<grid, block>>>(W, dY, dX);
        }else{
            linearXGradD<BM, BN, BK, RM, RN><<<grid, block>>>(W, dY, dX);
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return dX;
    }
    
    //[Unit Test Passed]
    Tensor* linearParamGrad(Tensor* dY, Tensor* X, Tensor* dW, Tensor* dB) {
        uint32 M = dW->dims.h;
        uint32 N = dW->dims.w;
        
        assert(X->dims.h == N);
        assert(dY->dims.h == M);
        assert(X->dims.n == dY->dims.n);
        
        dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        linearParamGradsD<BM, BN, BK, RM, RN><<<grid, block>>>(dY, X, dW, dB);
        cudaDeviceSynchronize();
        assertCuda(__FILE__,__LINE__);
        return dW;
    }
    
} // seblas