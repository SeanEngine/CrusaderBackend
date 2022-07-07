//
// Created by Dylan on 6/23/2022.
//

#include "Dropout.cuh"

namespace seann {
    void Dropout::initNetParams(OptimizerInfo *info, shape4 inShape) {
        mask = Tensor::create(inShape);
        X = Parameter::declare(inShape);
        Y = Parameter::create(inShape);
    }
    
    void Dropout::forward() {
        dropout(X->A, Y->A, mask, p);
    }
    
    void Dropout::xGrads() {
        *Y->dA + Y->dAReserve;
        dropoutGrad(X->dA, Y->dA, mask, p);
        Y->dAReserve->constFill(0);
    }
    
    string Dropout::info() {
        return "Dropout";
    }
    
    void Dropout::inferenceForward() {
        X->A->copyToD2D(Y->A);
        *Y->A * p;
    }
} // seann