//
// Created by Dylan on 6/20/2022.
//

#include "BatchNorm.cuh"

namespace seann {
    void BatchNorm::randFillNetParams() {
        beta->data()->constFill(0.0);
        gamma->data()->constFill(1.0);
    }
    
    void BatchNorm::forward() {
        batchNorm(X->A, beta->data(), gamma->data(), mean, variance, Y->A, xHatP);
    }
    
    void BatchNorm::xGrads() {
        *Y->dA + Y->dAReserve;
        batchNormGrad(Y->dA, gamma->data(), X->A, X->dA);
        Y->dAReserve->constFill(0);
    }
    
    void BatchNorm::paramGrads() {
        batchNormParamGrads(Y->dA, xHatP,  beta->grad(),gamma->grad());
    }
    
    void BatchNorm::updateParams() {
        beta->opt->apply();
        gamma->opt->apply();
    }
    
    void BatchNorm::batchUpdateParams() {
        beta->opt->batchApply();
        gamma->opt->batchApply();
    }
    
    string BatchNorm::info() {
        return "BatchNorm";
    }
    
    void BatchNorm::zeroGrads() {
        beta->opt->zeroGrad();
        gamma->opt->zeroGrad();
    }
    
    uint32 BatchNorm::encodeInfo(fstream *fout, uint64 offset) {
        return 0;
    }
    
    uint32 BatchNorm::encodeNetParams(fstream *fout, uint64 offset) {
        uint32 runningOffset = offset;
        runningOffset += beta->encodeNetParamData(fout, runningOffset);
        runningOffset += gamma->encodeNetParamData(fout, runningOffset);
        return runningOffset - offset;
    }
    
    OperandBase* DEC_OPR_BATCHNORM_INFO(fstream *fout, uint64& offset) {
        return new BatchNorm();
    }
    
   void DEC_OPR_BATCHNORM_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        auto* bn = (BatchNorm*)opr;
        bn->initNetParams(info, inShape);
        NetParam::decodeNetParamData(fin, offset, bn->beta);
        NetParam::decodeNetParamData(fin, offset, bn->gamma);
    }
} // seann