//
// Created by Dylan on 7/4/2022.
//

#include "ChannelConcatenater.cuh"
#include "../../../seblas/assist/Inspections.cuh"

namespace seann {
    void ChannelConcatenater::forward() {
        concat(Xs, paramCount, Y);
    }
    
    void ChannelConcatenater::xGrads() {
        *Y->dA + Y->dAReserve;
        concatGrads(Y, Xs, paramCount);
        cudaMemcpy(Xs[0]->dA->elements, Xs[0]->dAReserve->elements,
                   Xs[0]->dA->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
        Y->dAReserve->constFill(0);
        Xs[0]->dAReserve->constFill(0);
    }
} // seann