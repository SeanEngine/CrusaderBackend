//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_CULOSS_CUH
#define CRUSADER_CULOSS_CUH

#include "../tensor/Parameter.cuh"
#define YOLO_1_PREDICTORS_NUM 2
#define YOLO_1_CLASS_NUM 20
#define YOLO_1_CELL_NUM 7 * 7
#define YOLO_1_PARALLEL_CELLS 256

namespace seblas {
    typedef void(*LossFunc)(Parameter*, Tensor*);
    typedef float(*LossFuncCalc)(Parameter*, Tensor*, Tensor* buf);
    
    void crossEntropyLoss(Parameter* Y, Tensor* labels);
    
    float crossEntropyCalc(Parameter* Y, Tensor* labels, Tensor* buf);
    
    void Yolo1CompositeLoss(Parameter* Y, Tensor* labels);
    
    float Yolo1CompositeCalc(Parameter* Y, Tensor* labels, Tensor* buf);
} // seblas

#endif //CRUSADER_CULOSS_CUH
