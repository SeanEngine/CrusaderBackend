//
// Created by Dylan on 6/14/2022.
//

#ifndef CRUSADER_CUOPTIM_CUH
#define CRUSADER_CUOPTIM_CUH

#include "../tensor/Tensor.cuh"

namespace seblas {
    
    //applying SGD optimizer [UNP]
    void SGDApply(Tensor* X, Tensor* Xgrad, float LEARNING_RATE);
    
    //applying Momentum optimizer [UNP]
    void momentumApply(Tensor* X, Tensor* Xgrad, Tensor* V, float LEARNING_RATE, float MOMENTUM);
    
    //applying AdaGrad optimizer
    void adaGradApply(Tensor* X, Tensor* Xgrad, Tensor* V, float LEARNING_RATE, float EPSILON);
    
    //apply adaDelta (RMSProp) optimizer
    void adaDeltaApply(Tensor* X, Tensor* Xgrad, Tensor* V, Tensor* Vx, float RHO, float EPSILON);
    
    //applying Adam optimizer [UNP]
    void adamApply(Tensor* X, Tensor* Xgrad, Tensor* m, Tensor* V, float LEARNING_RATE,
                   float BETA1, float BETA2, float EPSILON, float t);
} // seblas

#endif //CRUSADER_CUOPTIM_CUH
