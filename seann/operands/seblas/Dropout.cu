//
// Created by Dylan on 6/23/2022.
//

#include "Dropout.cuh"

namespace seann {
    void Dropout::initNetParams(OptimizerInfo *info, shape4 inShape) {
        mask = Tensor::create(inShape);
        X = Parameter::declare(inShape);
        Y = Parameter::create(inShape);
    }
    
    void Dropout::forward() {
        dropout(X->A, Y->A, mask, p);
    }
    
    void Dropout::xGrads() {
        *Y->dA + Y->dAReserve;
        dropoutGrad(X->dA, Y->dA, mask, p);
        Y->dAReserve->constFill(0);
    }
    
    string Dropout::info() {
        return "Dropout";
    }
    
    void Dropout::inferenceForward() {
        X->A->copyToD2D(Y->A);
        *Y->A * p;
    }
    
    uint32 Dropout::encodeInfo(fstream *fout, uint64 offset) {
        fout->seekp(offset);
        fout->write((char*)&p, sizeof(float));
        return sizeof(float);
    }
    
    uint32 Dropout::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
    }
    
    OperandBase* DEC_OPR_DROPOUT_INFO(fstream* fin, uint64& offset) {
        float p;
        fin->seekg((long long)offset);
        fin->read((char*)&p, sizeof(float));
        offset += sizeof(float);
        return new Dropout(p);
    }
    
    void DEC_OPR_DROPOUT_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
} // seann