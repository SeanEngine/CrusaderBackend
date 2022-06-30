//
// Created by Dylan on 6/19/2022.
//

#include "Optimizers.cuh"

namespace seann {
    void SGD::apply() {
        if(isWeight) {
            *A->A * (1 - L2);
        }
        SGDApply(A->A, A->dA, LEARNING_RATE);
    }
    
    void SGD::zeroGrad() {}
    
    void BGD::batchApply() {
        if(isWeight) {
            *A->A * (1 - L2);
        }
        SGDApply(A->A, A->dA, LEARNING_RATE/BATCH_SIZE);
    }
    
    void BGD::zeroGrad() {}
    
    void Momentum::apply() {
        if(isWeight) {
            *A->A * (1 - L2);
        }
        momentumApply(A->A, A->dA, m, LEARNING_RATE, BETA);
    }
    
    void Momentum::zeroGrad() {
        m->constFill(0);
    }
    
    void AdaGrad::apply() {
        if(isWeight) {
            *A->A * (1 - L2);
        }
        adaGradApply(A->A, A->dA, V, LEARNING_RATE, EPSILON);
    }
    
    void AdaGrad::zeroGrad() {
        V->constFill(0);
    }
    
    void AdaDelta::apply() {
        if(isWeight) {
            *A->A * (1 - L2);
        }
        adaDeltaApply(A->A, A->dA, V, Vx, LEARNING_RATE, EPSILON);
    }
    
    void AdaDelta::zeroGrad() {
        V->constFill(0);
        Vx->constFill(0);
    }
    
    void Adam::apply() {
        if(isWeight) {
            *A->A * (1 - L2);
        }
        adamApply(A->A, A->dA, m, V, LEARNING_RATE, BETA1, BETA2, EPSILON, (float)t);
        t++;
    }
    
    void Adam::zeroGrad() {
        m->constFill(0);
        V->constFill(0);
    }
} // seann