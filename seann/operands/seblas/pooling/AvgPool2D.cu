//
// Created by Dylan on 7/10/2022.
//

#include "AvgPool2D.cuh"

namespace seann {
    string AvgPool2D::info() {
        return "AvgPool {" + X->A->dims.toString() + ", " + Y->A->dims.toString() + " }";
    }
    
    void AvgPool2D::xGrads() {
        *Y->dA + Y->dAReserve;
        Y->dAReserve->constFill(0);
        avgPoolBack(X->dA, Y->dA, strideH, strideW, rangeH, rangeW);
    }
    
    void AvgPool2D::forward() {
        avgPool(X->A, Y->A, strideH, strideW, rangeH, rangeW);
    }
} // seann