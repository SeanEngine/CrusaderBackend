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
    
    uint32 MaxPool2D::encodeInfo(fstream *fout, uint64 offset) {
        fout->seekp((long long) offset);
        fout->write((char*)&strideH, sizeof(uint32));
        fout->write((char*)&strideW, sizeof(uint32));
        fout->write((char*)&rangeH, sizeof(uint32));
        fout->write((char*)&rangeW, sizeof(uint32));
        return sizeof(uint32) * 4;
    }
    
    uint32 MaxPool2D::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
    }
    
    OperandBase* DEC_OPR_MAXPOOL2D_INFO(fstream* fin, uint64& offset) {
        uint32 strideH, strideW, rangeH, rangeW;
        fin->seekg((long long) offset);
        fin->read((char*)&strideH, sizeof(uint32));
        fin->read((char*)&strideW, sizeof(uint32));
        fin->read((char*)&rangeH, sizeof(uint32));
        fin->read((char*)&rangeW, sizeof(uint32));
        offset += sizeof(uint32) * 4;
        return new MaxPool2D(strideH, strideW, rangeH, rangeW);
    }
    
    void DEC_OPR_MAXPOOL2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
} // seamm