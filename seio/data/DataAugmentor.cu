//
// Created by Dylan on 6/27/2022.
//

#include "DataAugmentor.cuh"

namespace seio {
    
    void batchInit(Data** set, uint32 begin, uint32 miniBatchSize, Data* buf){
        uint32 indBeg = begin * miniBatchSize;
        for(uint32 i = 0; i <  miniBatchSize; i++) {
            cudaMemcpy(buf->X->elements + i * set[indBeg + i]->X->dims.size,
                       set[indBeg + i]->X->elements,
                       set[indBeg + i]->X->dims.size * sizeof(float),
                       cudaMemcpyHostToHost);
    
            cudaMemcpy(buf->label->elements + i * set[indBeg + i]->label->dims.size,
                       set[indBeg + i]->label->elements,
                       set[indBeg + i]->label->dims.size * sizeof(float),
                       cudaMemcpyHostToHost);
            assertCuda(__FILE__, __LINE__);
        }
    }
    
    Data *RandCorp::apply(Data* src) {
        uint32 hSpace = 2 * padH;
        uint32 wSpace = 2 * padW;
        default_random_engine generator(
                chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()
        );
        
        uniform_int_distribution<uint32> hDist(0, hSpace);
        uniform_int_distribution<uint32> wDist(0, wSpace);
        
        float* ptrSrc = src->X->elements;
        float* ptrDst = buf->X->elements;
        
        for (uint32 n = 0; n < miniBatchSize; n++){
            int hOffset = (int)hDist(generator) - (int)padH;
            int wOffset = (int)wDist(generator) - (int)padW;
            for (int c = 0; c < dataShape.c; c++){
                for (int h = 0; h < dataShape.h; h++){
                    for (int w = 0; w < dataShape.w; w++){
                        float val = h + hOffset > 0 && h + hOffset < dataShape.h &&
                                    w + wOffset > 0 && w + wOffset < dataShape.w ?
                                    ptrSrc[(h + hOffset) * src->X->dims.w + (w + wOffset)] : 0;
                        ptrDst[n * dataShape.size + c * dataShape.h * dataShape.w +
                            h * dataShape.w + w] = val;
                    }
                }
            }
        }
        
        cudaMemcpy(buf->label->elements, src->label->elements,
                   miniBatchSize * labelShape.size * sizeof(float),
                   cudaMemcpyHostToHost);
        assertCuda(__FILE__, __LINE__);
        return buf;
    }
    
    Data *RandFlipW::apply(Data *src) {
        float* ptrSrc = src->X->elements;
        float* ptrDst = buf->X->elements;
    
        default_random_engine generator(
                chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()
        );
        uniform_int_distribution<uint32> flipProp(0,1);
        
        for (uint32 n = 0; n < miniBatchSize; n++){
            if(flipProp(generator)) {
                for (int c = 0; c < dataShape.c; c++) {
                    for (int h = 0; h < dataShape.h; h++) {
                        for (int w = 0; w < dataShape.w; w++) {
                            float val = ptrSrc[n * dataShape.size + c * dataShape.h * dataShape.w +
                                               h * dataShape.w + w];
                            ptrDst[n * dataShape.size + c * dataShape.h * dataShape.w +
                                   h * dataShape.w + (dataShape.w - w)] = val;
                        }
                    }
                }
            } else {
                for (int c = 0; c < dataShape.c; c++) {
                    for (int h = 0; h < dataShape.h; h++) {
                        for (int w = 0; w < dataShape.w; w++) {
                            float val = ptrSrc[n * dataShape.size + c * dataShape.h * dataShape.w +
                                               h * dataShape.w + w];
                            ptrDst[n * dataShape.size + c * dataShape.h * dataShape.w +
                                   h * dataShape.w + w] = val;
                        }
                    }
                }
            }
        }
    
        cudaMemcpy(buf->label->elements, src->label->elements,
                   miniBatchSize * labelShape.size * sizeof(float),
                   cudaMemcpyHostToHost);
        assertCuda(__FILE__, __LINE__);
        return buf;
    }
}