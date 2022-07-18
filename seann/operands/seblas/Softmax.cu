//
// Created by Dylan on 6/20/2022.
//

#include "Softmax.cuh"

namespace seann {
    void Softmax::forward() {
        softmax(X->A, Y->A, reduceBuffer, Y->A->dims.size / Y->A->dims.n);
    }
    
    void Softmax::xGrads() {
        //Y->grad = Y->a - correct
        //this is controlled by loss function
        *Y->dA + Y->dAReserve;
        Y->dA->copyToD2D(X->dA);
        Y->dAReserve->constFill(0);
    }
    
    uint32 Softmax::encodeInfo(fstream *fout, uint64 offset) {
        return 0;
    }
    
    uint32 Softmax::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
    }
} // seann