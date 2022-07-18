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
    
    uint32 AvgPool2D::encodeInfo(fstream *fout, uint64 offset) {
        fout->seekp((long long) offset);
        fout->write((char*)&strideH, sizeof(uint32));
        fout->write((char*)&strideW, sizeof(uint32));
        fout->write((char*)&rangeH, sizeof(uint32));
        fout->write((char*)&rangeW, sizeof(uint32));
        return sizeof(uint32) * 4;
    }
    
    uint32 AvgPool2D::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
    }
} // seann