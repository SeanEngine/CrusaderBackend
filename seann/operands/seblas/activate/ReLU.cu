//
// Created by Dylan on 6/20/2022.
//

#include "ReLU.cuh"

namespace seann {
    void ReLU::forward() {
        relu(X->A, Y->A);
    }
    
    // ∂C/∂Z = ∂C/∂a * ∂a/∂Z = ∂C/∂a * σ'(Z)
    void ReLU::xGrads() {
        *Y->dA + Y->dAReserve;
        reluGradFast(X->A, Y->dA, X->dA);
        Y->dAReserve->constFill(0);
    }
    
    uint32 ReLU::encodeInfo(fstream *fout, uint64 offset) {
        return 0;
    }
    
    uint32 ReLU::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
    }
    
    OperandBase* DEC_OPR_RELU_INFO(fstream *fout, uint64& offset) {
        return new ReLU();
    }
    
    void DEC_OPR_RELU_PARAM(fstream *fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
} // seann