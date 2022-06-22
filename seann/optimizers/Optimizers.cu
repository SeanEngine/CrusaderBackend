//
// Created by Dylan on 6/19/2022.
//

#include "Optimizers.cuh"

namespace seann {
    void SGD::apply() {
        SGDApply(A->A, A->dA, LEARNING_RATE);
    }
    
    void SGD::zeroGrad() {}
    
    void BGD::batchApply() {
        SGDApply(A->A, A->dA, LEARNING_RATE/BATCH_SIZE);
    }
    
    void BGD::zeroGrad() {}
    
    void Momentum::apply() {
        momentumApply(A->A, A->dA, m, LEARNING_RATE, BETA);
    }
    
    void Momentum::zeroGrad() {
        m->constFill(0);
    }
    
    void AdaGrad::apply() {
        adaGradApply(A->A, A->dA, V, LEARNING_RATE, EPSILON);
    }
    
    void AdaGrad::zeroGrad() {
        V->constFill(0);
    }
    
    void AdaDelta::apply() {
        adaDeltaApply(A->A, A->dA, V, Vx, LEARNING_RATE, EPSILON);
    }
    
    void AdaDelta::zeroGrad() {
        V->constFill(0);
        Vx->constFill(0);
    }
    
    void Adam::apply() {
        adamApply(A->A, A->dA, m, V, LEARNING_RATE, BETA1, BETA2, EPSILON, (float)t);
        t++;
    }
    
    void Adam::zeroGrad() {
        m->constFill(0);
        V->constFill(0);
    }
} // seann