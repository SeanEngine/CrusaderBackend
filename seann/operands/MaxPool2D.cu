//
// Created by Dylan on 6/20/2022.
//

#include "MaxPool2D.cuh"

namespace seann {
    string MaxPool2D::info() {
        return "MaxPool2D     { " + X->A->dims.toString() + ", " + Y->A->dims.toString() + " }";
    }
    
    void MaxPool2D::forward() {
        maxPool(X->A, Y->A, record);
    }
    
    void MaxPool2D::xGrads() {
        maxPoolBack(X->dA, Y->dA, record);
    }
} // seamm