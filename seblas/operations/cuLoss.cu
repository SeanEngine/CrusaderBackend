//
// Created by Dylan on 6/17/2022.
//

#include "cuLoss.cuh"
#include "cuReduce.cuh"


namespace seblas {
    
    __device__ float calcIOU(float x, float y, float h, float w,
                            float lx, float ly, float lh, float lw){
        float x1 = x - w / 2;
        float y1 = y - h / 2;
        float x2 = x + w / 2;
        float y2 = y + h / 2;
        
        float lx1 = lx - lw / 2;
        float ly1 = ly - lh / 2;
        float lx2 = lx + lw / 2;
        float ly2 = ly + lh / 2;
        
        float intersectionX = max(0.0f, min(x2, lx2) - max(x1, lx1));
        float intersectionY = max(0.0f, min(y2, ly2) - max(y1, ly1));
        float area = h * w;
        float labelArea = lh * lw;
        float intersectionArea = intersectionX * intersectionY;
        return intersectionArea / (area + labelArea - intersectionArea);
    }
    
    __global__ void crossEntropyPrepare(Parameter* Y, Tensor* labels, Tensor* buf){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx < Y->A->dims.size ){
            float labelVal = labels->elements[idx];
            buf->elements[idx] = -log(Y->A->elements[idx] + 1e-10f) * labelVal;
        }
    }
    
    //each block will be taking care one of the channel
    //each thread will be taking care one of the cells
    //the label looks like this:
    
    //  [x] [y] [w] [h] [c] [class]
    template<const uint32 MAX_PARALLEL_CELLS, const uint32 CELL_COUNT,
            const uint32 BOX_COUNT, const uint32 CLASS_COUNT>
    __global__ void yolo1CompositeLossD(Parameter* Y, Tensor* labels,
                                        float lamda1 = 5, float lamda2 = 0.5){
        uint32 bID = blockIdx.x;
        uint32 readOffset = bID * CELL_COUNT * (BOX_COUNT * 5 + CLASS_COUNT);
        uint32 tid = threadIdx.x;
        
        const uint32 procCount = (CELL_COUNT / MAX_PARALLEL_CELLS) + 1;
        const uint32 elementSize = (BOX_COUNT * 5 + CLASS_COUNT);
        
        //register array for each thread
        //only contains the data for boxes since they will be used repeatingly
        float regisBoxes[BOX_COUNT * 5] = {0};
        float regisLabels[BOX_COUNT * 5] = {0};
        
        //main cycle, in case cellcount > blockDim.x
        #pragma unroll
        for(uint32 proc = 0; proc < procCount; proc++) {
            if(MAX_PARALLEL_CELLS * proc + tid >= CELL_COUNT) break;
            uint32 cellOffset = proc * MAX_PARALLEL_CELLS;
            
            //read box data from labels
            #pragma unroll BOX_COUNT * 5
            for (uint32 i = 0; i < BOX_COUNT * 5; i++) {
                regisLabels[i] = labels->elements[readOffset + cellOffset + tid * elementSize + i];
                regisBoxes[i] = Y->A->elements[readOffset + cellOffset + tid * elementSize + i];
            }
            
            //do the noobj loss if no object is detected
            if(regisLabels[4] == 0){
                #pragma unroll BOX_COUNT
                for (uint32 i = 0; i < BOX_COUNT; i++) {
                    Y->dA->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 4] =
                            lamda2 * (regisBoxes[i * 5 + 4] - regisLabels[i * 5 + 4]);
                }
                //skip the rest of the loop
                continue;
            }
            
            //calculate predictor with maximum iou
            float maxIou = 0;
            uint32 maxIouIdx = 0;
            #pragma unroll BOX_COUNT
            for (uint32 i = 0; i < BOX_COUNT; i++) {
                float iou = calcIOU(regisBoxes[i * 5 + 0], regisBoxes[i * 5 + 1],
                                    regisBoxes[i * 5 + 2], regisBoxes[i * 5 + 3],
                                    regisLabels[i * 5 + 0], regisLabels[i * 5 + 1],
                                    regisLabels[i * 5 + 2], regisLabels[i * 5 + 3]);
                if(iou > maxIou){
                    maxIou = iou;
                    maxIouIdx = i;
                }
            }
            
            //calculate the loss for the object
            #pragma unroll BOX_COUNT
            for (uint32 i = 0; i < BOX_COUNT; i++) {
                if(i == maxIouIdx){
                    //do the obj loss
                    // dx = x - x_hat
                    // dy = y - y_hat
                    Y->dA->elements[readOffset + cellOffset + tid * elementSize + i * 5] =
                            lamda1 * (regisBoxes[i * 5] - regisLabels[i * 5]);
                    Y->dA->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 1] =
                            lamda1 * (regisBoxes[i * 5 + 1] - regisLabels[i * 5 + 1]);
                    
                    // wl = 0.5 * (sqrt(w) - sqrt(w_hat))^2
                    // hl = 0.5 * (sqrt(h) - sqrt(h_hat))^2
                    // dw = (sqrt(w) - sqrt(w_hat)) / sqrt(w)
                    // dh = (sqrt(h) - sqrt(h_hat)) / sqrt(h)
                    Y->dA->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 2] =
                             lamda1 * (sqrtf(regisBoxes[i * 5 + 2]) - sqrtf(regisLabels[i * 5 + 2]))
                                      / (sqrtf(regisBoxes[i * 5 + 2]));
                    Y->dA->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 3] =
                             lamda1 * (sqrtf(regisBoxes[i * 5 + 3]) - sqrtf(regisLabels[i * 5 + 3]))
                                        / (sqrtf(regisBoxes[i * 5 + 3]));
                    
                    Y->dA->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 4] =
                            (regisBoxes[i * 5 + 4] - regisLabels[i * 5 + 4]);
                }else{
                    //do the noobj loss
                    Y->dA->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 4] =
                            lamda2 * (regisBoxes[i * 5 + 4] - regisLabels[i * 5 + 4]);
                }
            }
        }
    }
    
    template<const uint32 MAX_PARALLEL_CELLS, const uint32 CELL_COUNT,
            const uint32 BOX_COUNT, const uint32 CLASS_COUNT>
    __global__ void yolo1CompositePrepareD(Parameter* Y, Tensor* labels, Tensor* buf,
                                          float lamda1 = 5, float lamda2 = 0.5){
        uint32 bID = blockIdx.x;
        uint32 readOffset = bID * CELL_COUNT * (BOX_COUNT * 5 + CLASS_COUNT);
        uint32 tid = threadIdx.x;
    
        const uint32 procCount = (CELL_COUNT / MAX_PARALLEL_CELLS) + 1;
        const uint32 elementSize = (BOX_COUNT * 5 + CLASS_COUNT);
    
        //register array for each thread
        //only contains the data for boxes since they will be used repeatingly
        float regisBoxes[BOX_COUNT * 5] = {0};
        float regisLabels[BOX_COUNT * 5] = {0};
    
        //main cycle, in case cellcount > blockDim.x
        #pragma unroll
        for(uint32 proc = 0; proc < procCount; proc++) {
            if(MAX_PARALLEL_CELLS * proc + tid >= CELL_COUNT) break;
            uint32 cellOffset = proc * MAX_PARALLEL_CELLS;
        
            //read box data from labels
            #pragma unroll BOX_COUNT * 5
            for (uint32 i = 0; i < BOX_COUNT * 5; i++) {
                regisLabels[i] = labels->elements[readOffset + cellOffset + tid * elementSize + i];
                regisBoxes[i] = Y->A->elements[readOffset + cellOffset + tid * elementSize + i];
            }
        
            //do the noobj loss if no object is detected
            if(regisLabels[4] == 0){
                #pragma unroll BOX_COUNT
                for (uint32 i = 0; i < BOX_COUNT; i++) {
                    buf->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 4] =
                            lamda2 * powf((regisBoxes[i * 5 + 4] - regisLabels[i * 5 + 4]), 2);
                }
                //skip the rest of the loop
                continue;
            }
        
            //calculate predictor with maximum iou
            float maxIou = 0;
            uint32 maxIouIdx = 0;
            #pragma unroll BOX_COUNT
            for (uint32 i = 0; i < BOX_COUNT; i++) {
                float iou = calcIOU(regisBoxes[i * 5 + 0], regisBoxes[i * 5 + 1],
                                    regisBoxes[i * 5 + 2], regisBoxes[i * 5 + 3],
                                    regisLabels[i * 5 + 0], regisLabels[i * 5 + 1],
                                    regisLabels[i * 5 + 2], regisLabels[i * 5 + 3]);
                if(iou > maxIou){
                    maxIou = iou;
                    maxIouIdx = i;
                }
            }
        
            //calculate the loss for the object
            #pragma unroll BOX_COUNT
            for (uint32 i = 0; i < BOX_COUNT; i++) {
                if(i == maxIouIdx){
                    //do the obj loss
                    // dx = x - x_hat
                    // dy = y - y_hat
                    buf->elements[readOffset + cellOffset + tid * elementSize + i * 5] =
                            lamda1 * powf((regisBoxes[i * 5] - regisLabels[i * 5]),2);
                    buf->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 1] =
                            lamda1 * powf((regisBoxes[i * 5 + 1] - regisLabels[i * 5 + 1]),2);
                
                    // wl = 0.5 * (sqrt(w) - sqrt(w_hat))^2
                    // hl = 0.5 * (sqrt(h) - sqrt(h_hat))^2
                    // dw = (sqrt(w) - sqrt(w_hat)) / sqrt(w)
                    // dh = (sqrt(h) - sqrt(h_hat)) / sqrt(h)
                    buf->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 2] =
                            lamda1 * powf((sqrtf(regisBoxes[i * 5 + 2]) - sqrtf(regisLabels[i * 5 + 2])),2);
                    buf->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 3] =
                            lamda1 * powf((sqrtf(regisBoxes[i * 5 + 3]) - sqrtf(regisLabels[i * 5 + 3])),2);
    
                    buf->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 4] =
                            powf((regisBoxes[i * 5 + 4] - regisLabels[i * 5 + 4]),2);
                }else{
                    //do the noobj loss
                    buf->elements[readOffset + cellOffset + tid * elementSize + i * 5 + 4] =
                            lamda2 * powf((regisBoxes[i * 5 + 4] - regisLabels[i * 5 + 4]),2);
                }
            }
        }
    }
    
    float crossEntropyCalc(Parameter* Y, Tensor* label, Tensor* buf){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (buf->dims.size + block - 1) / block;
        
        crossEntropyPrepare<<<grid, block>>>(Y, label, buf);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        
        return reduce(buf, buf);
    }
    
    void crossEntropyLoss(Parameter* Y, Tensor* labels){
        Y->A->copyToD2D(Y->dA);
        *Y->dA - labels;
    }
    
    void Yolo1CompositeLoss(Parameter* Y, Tensor* labels){
        const uint32 block = YOLO_1_PARALLEL_CELLS;
        uint32 grid = Y->A->dims.n;
    
        yolo1CompositeLossD<YOLO_1_PARALLEL_CELLS, YOLO_1_CELL_NUM, YOLO_1_PREDICTORS_NUM
              , YOLO_1_CLASS_NUM><<<grid, block>>>(Y, labels);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    float Yolo1CompositeCalc(Parameter* Y, Tensor* labels, Tensor* buf){
        const uint32 block = YOLO_1_PARALLEL_CELLS;
        uint32 grid = Y->A->dims.n;
        buf->constFill(0);
    
        yolo1CompositePrepareD<YOLO_1_PARALLEL_CELLS, YOLO_1_CELL_NUM, YOLO_1_PREDICTORS_NUM
              , YOLO_1_CLASS_NUM><<<grid, block>>>(Y, labels, buf);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return reduce(buf, buf);
    }
} // seblas