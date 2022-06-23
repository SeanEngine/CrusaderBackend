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
        dropoutGrad(X->dA, Y->dA, mask, p);
    }
    
    string Dropout::info() {
        return "Dropout";
    }
} // seann