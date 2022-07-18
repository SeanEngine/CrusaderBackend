//
// Created by Dylan on 7/16/2022.
//

#include "LReLU.cuh"

namespace seann {
    void LReLU::forward() {
        lRelu(X->A, Y->A, alpha);
    }
    
    void LReLU::xGrads() {
        *Y->dA + Y->dAReserve;
        lReluGradFast(X->A, Y->dA, X->dA, alpha);
        Y->dAReserve->constFill(0);
    }
    
    uint32 LReLU::encodeInfo(fstream *fout, uint64 offset) {
        fout->seekp((long long) offset);
        fout->write((char*)&alpha, sizeof(float));
        return sizeof(float);
    }
    
    uint32 LReLU::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
    }
    
    OperandBase* DEC_OPR_LRELU_INFO(fstream* fin, uint64& offset) {
        float alpha;
        fin->seekg((long long) offset);
        fin->read((char*)&alpha, sizeof(float));
        offset += sizeof(float);
        return new LReLU(alpha);
    }
    
    void DEC_OPR_LRELU_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
} // seann