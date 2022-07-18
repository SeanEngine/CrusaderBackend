//
// Created by Dylan on 6/29/2022.
//

#include "ShortcutEndpoint.cuh"

namespace seann {
    void ShortcutEndpoint::initNetParams(OptimizerInfo *info, shape4 inShape) {
        
        X = Parameter::declare(inShape);
        Y = Parameter::create(inShape);
    }
    
    void ShortcutEndpoint::postWaiveInit(OptimizerInfo* inf) {
        if(isContainer) {
            //trace the other endpoint
            OperandBase* tracer = this->tracePrev();
            while(tracer->OPERAND_ID() != OPR_CTRL_SHORTCUTENDPOINT_BEG ||
                    ((ShortcutEndpoint*)tracer)->uuid != uuid) {
                tracer = tracer->tracePrev();
                assert(tracer != nullptr);
            }
            other = (ShortcutEndpoint*)tracer;
            assert(!other->isContainer);
            
            if(operandCount > 0) {
                branchOperands[0]->initNetParams(inf, other->Y->A->dims);
                for (auto i = 1; i < operandCount; i++) {
                    branchOperands[i]->initNetParams(inf, branchOperands[i - 1]->Y->A->dims);
                }
                assert(branchOperands[operandCount - 1]->Y->A->dims == Y->A->dims);
    
                branchOperands[0]->bindPrev(other);
                branchOperands[0]->bindInput(other->Y);
                branchOperands[0]->X->dA = other->Y->dAReserve;
    
                for (auto i = 1; i < operandCount; i++) {
                    branchOperands[i]->bindPrev(branchOperands[i - 1]);
                    branchOperands[i]->bindInput(branchOperands[i - 1]->Y);
                }
    
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i]->postWaiveInit(inf);
                }
            }
        }else{
            OperandBase* tracer = this->traceNext();
            while(tracer->OPERAND_ID() != OPR_CTRL_SHORTCUTENDPOINT_END ||
                  ((ShortcutEndpoint*)tracer)->uuid != uuid) {
                tracer = tracer->traceNext();
                assert(tracer != nullptr);
            }
            other = (ShortcutEndpoint*)tracer;
            assert(other->isContainer);
        }
    }
    
    void ShortcutEndpoint::forward() {
        //transport the main branch
        cudaMemcpy(Y->A->elements, X->A->elements,
                   X->A->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        if(isContainer) {
            //branch
            if(operandCount > 0) {
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i]->forward();
                }
                *Y->A + branchOperands[operandCount - 1]->Y->A;
            }else{
                //Identity shortcut
                *Y->A + other->Y->A;
            }
        }
    }
    
    void ShortcutEndpoint::xGrads() {
        if(isContainer) {
            if(operandCount > 0) {
                cudaMemcpy(branchOperands[operandCount-1]->Y->dA->elements, Y->dA->elements,
                           Y->A->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
                assertCuda(__FILE__, __LINE__);
    
                for (auto i = (int)operandCount-1; i >= 0; i--) {
                    branchOperands[i]->xGrads();
                }
            }else{
                //Identity shortcut
                cudaMemcpy(other->Y->dAReserve->elements, Y->dA->elements,
                           Y->A->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
                assertCuda(__FILE__, __LINE__);
            }
        }else{
            *Y->dA + Y->dAReserve;
            Y->dAReserve->constFill(0);
        }
        cudaMemcpy(X->dA->elements, Y->dA->elements,
                   X->A->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
    }
    
    void ShortcutEndpoint::paramGrads() {
        if(isContainer){
            if(operandCount > 0) {
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i]->paramGrads();
                }
            }
        }
    }
    
    void ShortcutEndpoint::updateParams() {
        if(isContainer){
            if(operandCount > 0) {
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i]->updateParams();
                }
            }
        }
    }
    
    void ShortcutEndpoint::batchUpdateParams() {
        if(isContainer){
            if(operandCount > 0) {
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i]->batchUpdateParams();
                }
            }
        }
    }
    
    void ShortcutEndpoint::randFillNetParams() {
        if(isContainer) {
            if(operandCount > 0) {
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i]->randFillNetParams();
                }
            }
        }
    }
    
    void ShortcutEndpoint::zeroGrads() {
        if(isContainer) {
            if(operandCount > 0) {
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i]->zeroGrads();
                }
            }
        }
    }
    
    uint32 ShortcutEndpoint::encodeInfo(fstream *fout, uint64 offset) {
        uint32 runningOffset = offset;
        fout->write((char *) &uuid, sizeof(uint32));
        fout->write((char *) &operandCount, sizeof(uint32));
        runningOffset += sizeof(uint32) * 2;
        if(isContainer){
            for (auto i = 0; i < operandCount; i++) {
                uint32 temp = (branchOperands[i]->OPERAND_ID());
                fout->write((char *) &temp, sizeof(uint32));
                runningOffset += sizeof(uint32);
            }
            for (auto i = 0; i < operandCount; i++) {
                runningOffset += branchOperands[i]->encodeInfo(fout, runningOffset);
            }
        }
        return runningOffset - offset;
    }
    
    uint32 ShortcutEndpoint::encodeNetParams(fstream *fout, uint64 offset) {
        uint32 runningOffset = offset;
        if(isContainer){
            for (auto i = 0; i < operandCount; i++) {
                runningOffset += branchOperands[i]->encodeNetParams(fout, runningOffset);
            }
        }
        return runningOffset - offset;
    }
} // seann