//
// Created by Dylan on 6/20/2022.
//

#include "Softmax.cuh"

namespace seann {
    void Softmax::forward() {
        softmax(X->A, Y->A, reduceBuffer, Y->A->dims.size / Y->A->dims.n);
    }
    
    void Softmax::xGrads() {
        //Y->grad = Y->a - correct
        //this is controlled by loss function
        Y->dA->copyToD2D(X->dA);
        *X->dA + X->dAReserve;
        X->dAReserve->constFill(0);
    }
} // seann