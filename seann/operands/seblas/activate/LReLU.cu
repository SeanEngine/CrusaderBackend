//
// Created by Dylan on 7/16/2022.
//

#include "LReLU.cuh"

namespace seann {
    void LReLU::forward() {
        lRelu(X->A, Y->A, alpha);
    }
    
    void LReLU::xGrads() {
        *Y->dA + Y->dAReserve;
        lReluGradFast(X->A, Y->dA, X->dA, alpha);
        Y->dAReserve->constFill(0);
    }
} // seann