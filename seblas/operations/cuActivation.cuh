//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_CUACTIVATION_CUH
#define CRUSADER_CUACTIVATION_CUH

#include "../tensor/Tensor.cuh"

namespace seblas {
    
    // relu activation
    Tensor* relu(Tensor* X, Tensor* Y);
    
    // relu derivative
    Tensor* reluGrad(Tensor* dY, Tensor* dX);
    
    Tensor* reluGradFast(Tensor* X, Tensor* dY, Tensor* dX);
    
    // lRelu activation
    Tensor* lRelu(Tensor* X, Tensor* Y, float alpha);
    
    // lRelu derivative
    Tensor* lReluGrad(Tensor* dY, Tensor* dX, float alpha);
    
    Tensor* lReluGradFast(Tensor* X, Tensor* dY, Tensor* dX, float alpha);
    
    // elu activation
    Tensor* elu(Tensor* X, Tensor* Y, float alpha);
    
    // elu derivative
    Tensor* eluGrad(Tensor* dY, Tensor* dX, float alpha);
    
    // sigmoid activation
    Tensor* sigmoid(Tensor* X, Tensor* Y);
    
    // sigmoid derivative
    Tensor* sigmoidGrad(Tensor* dY, Tensor* dX);
    
    // tanh activation
    Tensor* tanh(Tensor* X, Tensor* Y);
    
    // tanh derivative
    Tensor* tanhGrad(Tensor* dY, Tensor* dX);
    
} // seblas

#endif //CRUSADER_CUACTIVATION_CUH

