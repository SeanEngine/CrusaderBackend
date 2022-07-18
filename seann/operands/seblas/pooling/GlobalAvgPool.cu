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
    
    uint32 GlobalAvgPool::encodeInfo(fstream *fout, uint64 offset) {
        return 0;
    }
    
    uint32 GlobalAvgPool::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
    }
} // seann