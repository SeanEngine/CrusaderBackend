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
    
    OperandBase* DEC_OPR_AVGPOOL2D_INFO(fstream* fin, uint64& offset) {
        uint32 strideH, strideW, rangeH, rangeW;
        fin->seekg((long long)offset);
        fin->read((char*)&strideH, sizeof(uint32));
        fin->read((char*)&strideW, sizeof(uint32));
        fin->read((char*)&rangeH, sizeof(uint32));
        fin->read((char*)&rangeW, sizeof(uint32));
        offset += sizeof(uint32) * 4;
        return new AvgPool2D(strideH, strideW, rangeH, rangeW);
    }
    
    void DEC_OPR_AVGPOOL2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
} // seann