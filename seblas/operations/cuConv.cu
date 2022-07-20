
#include "cuConv.cuh"

#define BM 128
#define BN 128
#define BK 8
#define RM 8
#define RN 8

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

namespace seblas{
    /**
    * @brief The kernel for the convolution with 4D filters
    * @tparam BLOCK_M
    * @tparam BLOCK_N
    * @tparam BLOCK_K
    * @tparam REGIS_M
    * @tparam REGIS_N
    * @param A oc * ic * fh * fw
    * @param B on * ic * ih * iw
    * @param C on * oc * oh * ow
    * @param strideH
    * @param strideW
    * @param padH
    * @param padW
     *
     * [Unit test passed]
    */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmImplicit4D(Tensor* A, Tensor* B, Tensor* C, int strideH, int strideW, int padH, int padW,
            /*NULLABLE*/ Tensor* biases){
        
        // MatA: OC, IC * FH * FW; MatB: IC * FH * FW, OH * OW; Mat C: OC, OH * OW
        ///insert parameters
        const uint32 M = A->dims.n;
        const uint32 K = A->dims.c * A->dims.h * A->dims.w;
        const uint32 N = C->dims.n * C->dims.h * C->dims.w;
        
        const uint32 FH = A->dims.h;
        const uint32 FW = A->dims.w;
        const uint32 IC = B->dims.c;
        const uint32 IH = B->dims.h;
        const uint32 IW = B->dims.w;
        const uint32 OH = C->dims.h;
        const uint32 OW = C->dims.w;
        const uint32 OC = C->dims.c;
        
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
        float* ptrA = A->elements + blockIdx.y * BLOCK_M * K;
        float* ptrB = B->elements;
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
        
        #pragma unroll
        for(int i=0; i<BLOCK_M; i+= readRowStrideA){
            if(blockM + readRowA + i < M && readColA < K){
                tileA[0][readColA][readRowA+i] = ptrA[(readRowA + i)*K + readColA];
            }else{
                tileA[0][readColA][readRowA+i] = 0;
            }
        }
        
        ///this section is modified from its original state to suit the need for implicit gemm
        ///we are using A special mapping to instantiate patches as trajectories of conv filters
        #pragma unroll
        for(int i=0; i<BLOCK_K; i+= readRowStrideB){
            if(readRowB + i< K && blockN + readColB < N){
                
                //map buffer matrix cords to the 3 dimensional feature cords
                int in = (readColB + blockN) / (OH * OW);
                int oh = ((readColB + blockN) % (OH * OW))/OW;
                int ow = ((readColB + blockN) % (OH * OW))%OW;
                int ic = (readRowB + i)/(FH * FW);
                int fh = ((readRowB + i)%(FH * FW))/FW;
                int fw = ((readRowB + i)%(FH * FW))%FW;
                int ih = oh * strideH - padH + fh;
                int iw = ow * strideW - padW + fw;
                //do memory access
                tileB[0][readRowB+i][readColB] = ih >= 0 && iw >= 0 && ih < IH && iw < IW ?
                                                 ptrB[in * IC * IH * IW + ic * IH * IW + ih * IW + iw] : 0;
            }else{
                tileB[0][readRowB+i][readColB] = 0;
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
                                         ptrA[(readRowA + i) * K + readColA + nextTileID] : 0;
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                    
                    //calculate remapping
                    int loadIndex = i / readRowStrideB;
                    int in = (readColB + blockN) / (OH * OW);
                    int oh = ((readColB + blockN) % (OH * OW))/OW;
                    int ow = ((readColB + blockN) % (OH * OW))%OW;
                    int ic = (readRowB + i + nextTileID)/(FH * FW);
                    int fh = ((readRowB + i + nextTileID)%(FH * FW))/FW;
                    int fw = ((readRowB + i + nextTileID)%(FH * FW))%FW;
                    int ih = oh * strideH - padH + fh;
                    int iw = ow * strideW - padW + fw;
                    
                    //do memory access
                    bufferB[loadIndex] = (readRowB + i + nextTileID < K && blockN + readColB < N) && (ih >= 0 && iw >= 0)
                                         && (ih < IH && iw < IW)? ptrB[in * IC * IH * IW + ic * IH * IW + ih * IW + iw] : 0;
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
                    float bias = biases == nullptr ? 0 : biases->elements[blockM + threadIdx.y * REGIS_M + rm];
                    //remapping is needed since the original output matrix would be ( OC, OH * OW * ON )
                    uint32 on = (blockN + threadIdx.x * REGIS_N + rn) / (OH * OW);
                    C->elements[on * OC * OH * OW + (blockM + threadIdx.y * REGIS_M + rm) * (OH * OW)
                                + (blockN + threadIdx.x * REGIS_N + rn)%(OH * OW)] = regisC[rm][rn] + bias;
                }
            }
        }
    }
    
    /**
     * @brief The back propagation of conv layers (relative to input features)
     * @tparam BLOCK_M
     * @tparam BLOCK_N
     * @tparam BLOCK_K
     * @tparam REGIS_M
     * @tparam REGIS_N
     * @param A OC * IC * FH * FW
     * @param B ON * OC * OH * OW
     * @param C ON * IC * IH * IW
     * @param strideH original input feature's strideH, but stands for upsampling factor
     * @param strideW original input feature's strideW, but stands for upsampling factor
     * @param newPadH pre-processed padding (different from original forward calculation)
     * @param newPadW pre-processed padding (different from original forward calculation)
     */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmImplicitBackprop(Tensor* A, Tensor* B, Tensor* C,
                 int strideH, int strideW, int newPadH, int newPadW){
        ///insert parameters
        //order changed for reversed convolution
        const uint32 M = A->dims.c;
        const uint32 K = A->dims.n * A->dims.h * A->dims.w;
        const uint32 SRC_K = A->dims.c * A->dims.h * A->dims.w;
        const uint32 N = C->dims.n * C->dims.h * C->dims.w;
        
        const uint32 FH = A->dims.h;
        const uint32 FW = A->dims.w;
        const uint32 IC = B->dims.c;
        const uint32 IH = B->dims.h;
        const uint32 IW = B->dims.w;
        const uint32 OH = C->dims.h;
        const uint32 OW = C->dims.w;
        const uint32 OC = C->dims.c;
        
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
        
        #pragma unroll
        for(int i=0; i<BLOCK_M; i+= readRowStrideA){
            int rowFilterIndex = readColA / (FH * FW);
            int colFilterIndex = blockM + readRowA + i;
            int colIndex = (FH * FW - 1) - (readColA % (FH * FW));
            if(blockM + readRowA + i < M && readColA < K){
                //rotate 180 degrees
                tileA[0][readColA][readRowA+i] = ptrA[rowFilterIndex * SRC_K + colFilterIndex * (FH * FW) + colIndex];
            } else {
                tileA[0][readColA][readRowA+i] = 0;
            }
        }
        
        ///this section is modified from its original state to suit the need for implicit gemm
        ///we are using A special mapping to instantiate patches as trajectories of conv filters
        #pragma unroll
        for(int i=0; i<BLOCK_K; i+= readRowStrideB){
            if(readRowB + i< K && blockN + readColB < N){
                
                //map buffer matrix cords to the 3 dimensional feature cords
                int in = (readColB + blockN) / (OH * OW);
                int oh = ((readColB + blockN) % (OH * OW))/OW;
                int ow = ((readColB + blockN) % (OH * OW))%OW;
                int ic = (readRowB + i)/(FH * FW);
                int fh = ((readRowB + i)%(FH * FW))/FW;
                int fw = ((readRowB + i)%(FH * FW))%FW;
                int ih = oh - newPadH + fh;
                int iw = ow - newPadW + fw;
                int procIh = ih / strideH;
                int procIw = iw / strideW;
                //do memory access
                tileB[0][readRowB+i][readColB] = procIh >= 0 && procIw >= 0 && procIh < IH &&
                                                 procIw < IW && ih % strideH == 0 && iw % strideW == 0 ?
                                                 ptrB[in * IC * IH * IW + ic * IH * IW + procIh * IW + procIw] : 0;
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
                    int rowFilterIndex = (readColA + nextTileID) / (FH * FW);
                    int colFilterIndex = blockM + readRowA + i;
                    int colIndex = (FH * FW - 1) - ((readColA + nextTileID) % (FH * FW));
                    bufferA[loadIndex] = blockM + readRowA + i < M && readColA + nextTileID < K ?
                                         ptrA[rowFilterIndex * SRC_K + colFilterIndex * (FH * FW) + colIndex] : 0;
                }
                
                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                    
                    //calculate remapping
                    int loadIndex = i / readRowStrideB;
                    int in = (readColB + blockN) / (OH * OW);
                    int oh = ((readColB + blockN) % (OH * OW))/OW;
                    int ow = ((readColB + blockN) % (OH * OW))%OW;
                    int ic = (readRowB + i + nextTileID)/(FH * FW);
                    int fh = ((readRowB + i + nextTileID)%(FH * FW))/FW;
                    int fw = ((readRowB + i + nextTileID)%(FH * FW))%FW;
                    int ih = oh - newPadH + fh;
                    int iw = ow - newPadW + fw;
                    int procIh = ih / strideH;
                    int procIw = iw / strideW;
                    
                    //do memory access
                    bufferB[loadIndex] = (readRowB + i + nextTileID < K && blockN + readColB < N) && (procIh >= 0 && procIw >= 0)
                                         && (procIh < IH && procIw < IW) && ih % strideH == 0 && iw % strideW == 0 ?
                                         ptrB[in * IC * IH * IW + ic * IH * IW + procIh * IW + procIw] : 0;
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
                    //remapping is needed since the original output matrix would be ( OC, OH * OW * ON )
                    uint32 on = (blockN + threadIdx.x * REGIS_N + rn) / (OH * OW);
                    C->elements[on * OC * OH * OW + (blockM + threadIdx.y * REGIS_M + rm) * (OH * OW)
                                + (blockN + threadIdx.x * REGIS_N + rn)%(OH * OW)] = regisC[rm][rn];
                }
            }
        }
    }
    
    ///[Unit Test Passed]
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmImplicitError(Tensor* A, Tensor* B, Tensor* C, int strideH, int strideW, int padH, int padW){
        
        ///insert parameters
        const uint32 M = A->dims.c;
        const uint32 K = A->dims.h * A->dims.w * A->dims.n;
        const uint32 N = C->dims.c * C->dims.h * C->dims.w;

        const uint32 FH = A->dims.h;
        const uint32 FW = A->dims.w;
        const uint32 FC = A->dims.c;
        const uint32 IC = B->dims.c;
        const uint32 IH = B->dims.h;
        const uint32 IW = B->dims.w;
        const uint32 OH = C->dims.h;
        const uint32 OW = C->dims.w;

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

        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
            if (blockM + readRowA + i < M && readColA < K) {
                int procReadNDim = readColA / (FH * FW);
                int procReadCol = readColA % (FH * FW);
                tileA[0][readColA][readRowA + i] = ptrA[procReadNDim * FH * FW * FC +
                                     (blockM + readRowA + i) * FH * FW + procReadCol];
            } else {
                tileA[0][readColA][readRowA + i] = 0;
            }
        }

        ///this section is modified from its original state to suit the need for implicit gemm
        ///we are using A special mapping to instantiate patches as trajectories of conv filters
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
            if (readRowB + i < K && blockN + readColB < N) {

                //map buffer matrix cords to the 3 dimensional feature cords
                int ic = (readColB + blockN) / (OH * OW);
                int oh = ((readColB + blockN) % (OH * OW)) / OW;
                int ow = ((readColB + blockN) % (OH * OW)) % OW;

                int in = (readRowB + i) / (FH * FW);
                int rst = (readRowB + i) % (FH * FW);
                int fh = ((rst) % (int)(FH * FW)) / (int)FW;
                int fw = ((rst) % (int)(FH * FW)) % (int)FW;
                int ih = oh * strideH - padH + fh * strideH;
                int iw = ow * strideW - padW + fw * strideW;
                //do memory access
                tileB[0][readRowB + i][readColB] = ih >= 0 && iw >= 0 && ih < IH && iw < IW ?
                                                   ptrB[in * IC * IH * IW + ic * IH * IW + ih * IW + iw] : 0;
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
                for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                    int loadIndex = i / readRowStrideA;
                    int procReadNDim = (readColA + nextTileID) / (FH * FW);
                    int procReadCol = (readColA + nextTileID) % (FH * FW);
                    bufferA[loadIndex] = blockM + readRowA + i < M && readColA + nextTileID < K ?
                                         ptrA[procReadNDim * FH * FW * FC +
                                                 (blockM + readRowA + i) * FH * FW + procReadCol] : 0;
                }

                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideB) {

                    //calculate remapping
                    int loadIndex = i / readRowStrideB;
                    int ic = (readColB + blockN) / (OH * OW);
                    int oh = ((readColB + blockN) % (OH * OW)) / OW;
                    int ow = ((readColB + blockN) % (OH * OW)) % OW;

                    int in = (readRowB + i + nextTileID) / (FH * FW);
                    int rst = (readRowB + i + nextTileID) % (FH * FW);
                    int fh = (rst % (int)(FH * FW)) / (int)FW;
                    int fw = (rst % (int)(FH * FW)) % (int)FW;
                    int ih = oh * strideH - padH + fh * strideH;
                    int iw = ow * strideW - padW + fw * strideW;

                    //do memory access
                    bufferB[loadIndex] =
                            (readRowB + i + nextTileID < K && blockN + readColB < N) && (ih >= 0 && iw >= 0)
                            && (ih < IH && iw < IW) ? ptrB[in * IC * IH * IW + ic * IH * IW + ih * IW + iw] : 0;
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
        for(int rm = 0; rm < REGIS_M; rm ++){
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn ++){
                if((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                + (blockN + threadIdx.x * REGIS_N + rn)] += regisC[rm][rn];
                }
            }
        }
    }
    
    __global__ void convBiasD(Tensor* Y, Tensor* bias){
        uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32 ndim = blockIdx.y;
        uint32 globalWarpID = tid / WARP_SIZE;
        uint32 laneID = threadIdx.x % WARP_SIZE;
        
        //each warp will be covering 1 channel from the input tensor
        uint32 readRange = Y->dims.h * Y->dims.w;
        uint32 readBeg = laneID * (readRange / WARP_SIZE);
        uint32 readEnd = laneID == WARP_SIZE - 1 ? readRange : readBeg + readRange / WARP_SIZE;
        if(globalWarpID >= bias->dims.size) return;
    
        float biasVal = bias->elements[globalWarpID];
        #pragma unroll
        for(uint32 i = readBeg; i < readEnd; i ++){
            float val = Y->elements[ndim * (Y->dims.size / Y->dims.n) +
                                    globalWarpID * readRange + i];
            val = val + biasVal;
            Y->elements[ndim * (Y->dims.size / Y->dims.n) + globalWarpID * readRange + i] = val;
        }
    }
    
    void assertConv(Tensor* filters, Tensor* features, Tensor* featureOut, uint32 strideH, uint32 strideW, uint32 padH, uint32 padW){
        if(featureOut->dims.h != (features->dims.h - filters->dims.h + 2 * padH) / strideH + 1){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: rows (h) relationship did not satisfy");
            throw std::invalid_argument("assertConv: rows (h) relationship did not satisfy");
        }
        
        if(featureOut->dims.w != (features->dims.w - filters->dims.w + 2 * padW) / strideW + 1){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: cols (w) relationship did not satisfy");
            throw std::invalid_argument("assertConv: cols (w) relationship did not satisfy");
        }
        
        if(featureOut->dims.c != filters->dims.n || features->dims.c != filters->dims.c){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: channels (c) relationship did not satisfy");
            throw std::invalid_argument("assertConv: channels relationship did not satisfy");
        }
        
        if(featureOut->dims.n != features->dims.n){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: batch (n) relationship did not satisfy");
            throw std::invalid_argument("assertConv: batch relationship did not satisfy");
        }
    }
    
    Tensor* conv(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW, Tensor* biases) {
        assertConv(A,B,C, strideH, strideW, padH, padW);
        
        uint32 M = A->dims.n;
        uint32 N = C->dims.h * C->dims.w * C->dims.n;
        
        dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        gemmImplicit4D<BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C, strideH, strideW, padH, padW, biases);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return C;
    }
    
    
    //C is the errors of prev layer and B is for this layer
    Tensor* convDerive(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW) {
        assertConv(A, C, B, strideH, strideW, padH, padW);
        
        uint32 M = A->dims.c;
        uint32 N = C->dims.h * C->dims.w * C->dims.n;
        
        int newPadH = (int)(C->dims.h + A->dims.h - 1 - B->dims.h * strideH)/2;
        int newPadW = (int)(C->dims.w + A->dims.w - 1 - B->dims.w * strideH)/2;
        
        dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        gemmImplicitBackprop<BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C, strideH, strideW, newPadH, newPadW);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return C;
    }
    
    //A is the error of layer, B is the input, C is the filters grads
    Tensor* convError(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW) {
        assertConv(C, B, A, strideH, strideW, padH, padW);
        
        uint32 M = A->dims.c;
        uint32 N = C->dims.h * C->dims.w * C->dims.c;
        
        dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);
        
        gemmImplicitError<BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C, strideH, strideW, padH, padW);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return A;
    }
    
    Tensor* convBias(Tensor* Y, Tensor* bias){
        assert(bias->dims.size == Y->dims.c);
        uint32 warps = (CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y) / WARP_SIZE;
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        dim3 grid = dim3((bias->dims.size + warps - 1) / warps,
                         Y->dims.n);
        convBiasD<<<grid, block>>>(Y, bias);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return Y;
    }
}