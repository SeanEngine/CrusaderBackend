//
// Created by Dylan on 7/4/2022.
//

#include "ChannelConcatenater.cuh"

namespace seann {
    void ChannelConcatenater::forward() {
        concat(Xs, paramCount, Y);
    }
    
    void ChannelConcatenater::xGrads() {
        *Y->dA + Y->dAReserve;
        concatGrads(Y, Xs, paramCount);
        Y->dAReserve->constFill(0);
    }
} // seann