//
// Created by Dylan on 6/20/2022.
//

#include "Linear.cuh"

namespace seann {
    void Linear::initNetParams(OptimizerInfo *info, shape4 inShape) {
        INPUT_SIZE = inShape.size/inShape.n;
        X = Parameter::declare(inShape.n , 1, INPUT_SIZE, 1);
        Y = Parameter::create(inShape.n, 1, OUTPUT_SIZE, 1);
        weights = (new NetParam(info, OUTPUT_SIZE, INPUT_SIZE))->setWeight();
        biases = new NetParam(info, OUTPUT_SIZE, 1);
    }
    
    // a[l] = w[l] * a[l-1] + b[l]
    void Linear::forward() {
        linear(X->A, weights->data(), biases->data(), Y->A);
    }
    
    void Linear::paramGrads() {
        linearParamGrad(Y->dA, X->A, weights->grad(), biases->grad());
    }
    
    void Linear::updateParams() {
        weights->opt->apply();
        biases->opt->apply();
    }
    
    void Linear::batchUpdateParams() {
        weights->opt->batchApply();
        biases->opt->batchApply();
    }
    
    void Linear::xGrads() {
        // ∂x = w^T * ∂z
        *Y->dA + Y->dAReserve;
        linearXGrad(Y->dA, weights->data(), X->dA);
        Y->dAReserve->constFill(0);
    }
    
    void Linear::randFillNetParams() {
        uint32 K = weights->data()->dims.w;
        weights->data()->randNormal( 0, (float)sqrt(2.0 / (float) K));
        biases->data()->randNormal(0, (float)sqrt(2.0 / (float) biases->data()->dims.size));
    }
    
    void Linear::zeroGrads() {
        weights->opt->zeroGrad();
        biases->opt->zeroGrad();
    }
    
    float Linear::getOptimLR() {
        return weights->opt->LEARNING_RATE;
    }
    
    void Linear::updateOptimLR(float val) {
        weights->opt->LEARNING_RATE = val;
        biases->opt->LEARNING_RATE = val;
    }
} // seann