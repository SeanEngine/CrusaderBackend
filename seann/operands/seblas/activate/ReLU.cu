//
// Created by Dylan on 6/20/2022.
//

#include "ReLU.cuh"

namespace seann {
    void ReLU::forward() {
        relu(X->A, Y->A);
    }
    
    // ∂C/∂Z = ∂C/∂a * ∂a/∂Z = ∂C/∂a * σ'(Z)
    void ReLU::xGrads() {
        *Y->dA + Y->dAReserve;
        reluGradFast(X->A, Y->dA, X->dA);
        Y->dAReserve->constFill(0);
    }
} // seann