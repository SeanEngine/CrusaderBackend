//
// Created by Dylan on 6/20/2022.
//

#include "MaxPool2D.cuh"

namespace seann {
    string MaxPool2D::info() {
        return "MaxPool2D     { " + X->A->dims.toString() + ", " + Y->A->dims.toString() + " }";
    }
    
    void MaxPool2D::forward() {
        maxPool(X->A, Y->A, record, strideH, strideW, rangeH, rangeW);
    }
    
    void MaxPool2D::xGrads() {
        *Y->dA + Y->dAReserve;
        maxPoolBack(X->dA, Y->dA, record, strideH, strideW, rangeH, rangeW);
        Y->dAReserve->constFill(0);
    }
} // seamm