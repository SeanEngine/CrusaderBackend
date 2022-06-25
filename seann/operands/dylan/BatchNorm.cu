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
        batchNorm(X->A, beta->data(), gamma->data(), mean, variance, Y->A);
    }
    
    void BatchNorm::xGrads() {
        batchNormGrad(Y->dA, gamma->data(), X->A, X->dA);
    }
    
    void BatchNorm::paramGrads() {
        batchNormParamGrads(Y->dA,  beta->grad(),gamma->grad(), X->A);
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