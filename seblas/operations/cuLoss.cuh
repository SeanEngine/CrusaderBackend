//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_CULOSS_CUH
#define CRUSADER_CULOSS_CUH

#include "../tensor/Parameter.cuh"

namespace seblas {
    typedef void(*LossFunc)(Parameter*, Tensor*);
    typedef float(*LossFuncCalc)(Parameter*, Tensor*, Tensor* buf);
    
    void crossEntropyLoss(Parameter* Y, Tensor* labels);
    
    float crossEntropyCalc(Parameter* Y, Tensor* labels, Tensor* buf);
    
    void Yolo1CompositeLoss(Parameter* Y, Tensor* labels);
    
    float Yolo1CompositeCalc(Parameter* Y, Tensor* labels, Tensor* buf);
} // seblas

#endif //CRUSADER_CULOSS_CUH
