//
// Created by Dylan on 6/27/2022.
//

#include "DataTransform.cuh"

Tensor *DistribNorm::apply(Tensor *X) {
    uint32 step = X->dims.size / X->dims.c;
    for(uint32 c = 0; c < X->dims.c; c++){
        float srcMean = 0;
        float srcVar = 0;
        for(uint32 i = 0; i < step; i++){
            srcMean += X->elements[c * step + i];
        }
        srcMean /= (float)step;
        for(uint32 i = 0; i < step; i++){
            srcVar += (X->elements[c * step+i] - srcMean) * (X->elements[c * step+i] - srcMean);
        }
        srcVar /= (float)step;
        
        for(uint32 i = 0; i < step; i++){
            X->elements[c * step+i] = (X->elements[c * step+i] - srcMean) * desiredVar
                    / (float)sqrt(srcVar + 1e-8) + desiredMean;
        }
    }
    return X;
}

Tensor *UniformNorm::apply(Tensor *X) {
    uint32 step = X->dims.size / X->dims.c;
    float max = -1e10;
    float min = 1e10;
    for(uint32 c = 0; c < X->dims.c; c++){
        for(uint32 i = 0; i < step; i++){
            if(X->elements[c * step+i] > max) max = X->elements[c * step+i];
            if(X->elements[c * step+i] < min) min = X->elements[c * step+i];
        }
        
        for(uint32 i = 0; i < step; i++){
            X->elements[c * step+i] = (X->elements[c * step+i] - min) / (max - min) * (desiredMax - desiredMin) + desiredMin;
        }
    }
    return X;
}
