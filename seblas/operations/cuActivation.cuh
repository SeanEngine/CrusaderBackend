//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_CUACTIVATION_CUH
#define CRUSADER_CUACTIVATION_CUH

#include "../tensor/Tensor.cuh"

namespace seblas {
    
    // relu activation
    Tensor* relu(Tensor* X, Tensor* outX);
    
    // relu derivative
    Tensor* reluGrad(Tensor* X, Tensor* outX);
    
    Tensor* reluGradFast(Tensor* Z, Tensor* dY, Tensor* dZ);
    
    // lRelu activation
    Tensor* lRelu(Tensor* X, Tensor* outX, float alpha);
    
    // lRelu derivative
    Tensor* lReluGrad(Tensor* X, Tensor* outX, float alpha);
    
    // elu activation
    Tensor* elu(Tensor* X, Tensor* outX, float alpha);
    
    // elu derivative
    Tensor* eluGrad(Tensor* X, Tensor* outX, float alpha);
    
    // sigmoid activation
    Tensor* sigmoid(Tensor* X, Tensor* outX);
    
    // sigmoid derivative
    Tensor* sigmoidGrad(Tensor* X, Tensor* outX);
    
    // tanh activation
    Tensor* tanh(Tensor* X, Tensor* outX);
    
    // tanh derivative
    Tensor* tanhGrad(Tensor* X, Tensor* outX);
    
} // seblas

#endif //CRUSADER_CUACTIVATION_CUH

