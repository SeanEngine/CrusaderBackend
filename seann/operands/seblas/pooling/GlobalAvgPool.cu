//
// Created by Dylan on 7/16/2022.
//

#include "GlobalAvgPool.cuh"

namespace seann {
    string GlobalAvgPool::info() {
        return "GlobalAvgPool";
    }
    
    void GlobalAvgPool::forward() {
        globalAvgPool(X->A, Y->A, buffer);
    }
    
    void GlobalAvgPool::xGrads() {
        *Y->dA + Y->dAReserve;
        Y->dAReserve->constFill(0);
        globalAvgPoolBack(X->dA, Y->dA);
    }
} // seann