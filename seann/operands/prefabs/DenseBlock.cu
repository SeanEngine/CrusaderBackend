//
// Created by Dylan on 7/4/2022.
//

#include "DenseBlock.cuh"
#include "../../../seblas/assist/Inspections.cuh"

namespace seann {
    void DenseBlock::forward() {
        for(uint32 i = 0; i < operandCount; i++) {
            operands[i]->forward();
        }
    
        cudaMemcpy(Y->A->elements, operands[operandCount-1]->Y->A->elements,
                   Y->A->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
    }
    
    void DenseBlock::xGrads() {
        *Y->dA + Y->dAReserve;
        cudaMemcpy(operands[operandCount-1]->Y->dA->elements, Y->dA->elements,
                   Y->A->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
        for(int i = (int)operandCount-1; i >= 0 ; i--) {
            operands[i]->xGrads();
        }
        Y->dAReserve->constFill(0);
    }
    
    void DenseBlock::paramGrads() {
        for(uint32 i = 0; i < operandCount; i++) {
            operands[i]->paramGrads();
        }
    }
    
    void DenseBlock::updateParams() {
        for(uint32 i = 0; i < operandCount; i++) {
            operands[i]->updateParams();
        }
    }
    
    void DenseBlock::batchUpdateParams() {
        for(uint32 i = 0; i < operandCount; i++) {
            operands[i]->batchUpdateParams();
        }
    }
    
    void DenseBlock::zeroGrads() {
        for(uint32 i = 0; i < operandCount; i++) {
            operands[i]->zeroGrads();
        }
    }
    
    string DenseBlock::info() {
        return "DenseBlock: { l=" + std::to_string(l) + ", k="+ std::to_string(k)+ " }";
    }
    
    void DenseBlock::randFillNetParams() {
        for(uint32 i = 0; i < operandCount; i++) {
            operands[i]->randFillNetParams();
        }
    }
    
    uint32 DenseBlock::encodeInfo(fstream *fout, uint64 offset) {
        uint32 runningOffset = offset;
        fout->seekp((long long) offset);
        fout->write((char*)&l, sizeof(uint32));
        fout->write((char*)&k, sizeof(uint32));
        runningOffset += sizeof(uint32) * 2;
        return runningOffset - offset;
    }
    
    uint32 DenseBlock::encodeNetParams(fstream *fout, uint64 offset) {
        uint32 runningOffset = offset;
        for(uint32 i = 0; i < operandCount; i++) {
            runningOffset += operands[i]->encodeNetParams(fout, runningOffset);
        }
        return runningOffset - offset;
    }
} // seann