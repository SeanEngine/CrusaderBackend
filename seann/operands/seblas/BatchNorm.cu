//
// Created by Dylan on 6/20/2022.
//

#include "BatchNorm.cuh"

namespace seann {
    void BatchNorm::randFillNetParams() {
        beta->data()->constFill(0.0);
        gamma->data()->constFill(1.0);
    }
    
    void BatchNorm::forward() {
        batchNorm(X->A, beta->data(), gamma->data(), mean, variance, Y->A, xHatP);
    }
    
    void BatchNorm::xGrads() {
        *Y->dA + Y->dAReserve;
        batchNormGrad(Y->dA, gamma->data(), X->A, X->dA);
        Y->dAReserve->constFill(0);
    }
    
    void BatchNorm::paramGrads() {
        batchNormParamGrads(Y->dA, xHatP,  beta->grad(),gamma->grad());
    }
    
    void BatchNorm::updateParams() {
        beta->opt->apply();
        gamma->opt->apply();
    }
    
    void BatchNorm::batchUpdateParams() {
        beta->opt->batchApply();
        gamma->opt->batchApply();
    }
    
    string BatchNorm::info() {
        return "BatchNorm";
    }
    
    void BatchNorm::zeroGrads() {
        beta->opt->zeroGrad();
        gamma->opt->zeroGrad();
    }
} // seann