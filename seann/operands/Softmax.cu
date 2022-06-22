//
// Created by Dylan on 6/20/2022.
//

#include "Softmax.cuh"

namespace seann {
    void seann::Softmax::forward() {
        softmax(X->A, Y->A, reduceBuffer, Y->A->dims.size / Y->A->dims.n);
    }
    
    void seann::Softmax::xGrads() {
        //Y->grad = Y->a - correct
        //this is controlled by loss function
        Y->dA->copyToD2D(X->dA);
    }
} // seann