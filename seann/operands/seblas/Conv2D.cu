//
// Created by Dylan on 6/20/2022.
//

#include "Conv2D.cuh"

namespace seann {
    void Conv2D::forward() {
        if(WITH_BIAS){
            conv(filter->data(), X->A, Y->A,
                 (int)strideH, (int)strideW, (int)padH, (int)padW, bias->data());
            return;
        }
        conv(filter->data(), X->A, Y->A,
             (int)strideH, (int)strideW, (int)padH, (int)padW, nullptr);
    }
    
    void Conv2D::xGrads() {
        *Y->dA + Y->dAReserve;
        convDerive(filter->data(), Y->dA, X->dA, (int)strideH, (int)strideW, (int)padH, (int)padW);
        Y->dAReserve->constFill(0);
    }
    
    void Conv2D::paramGrads() {
        convError(Y->dA, X->A, filter->grad(), (int)strideH, (int)strideW, (int)padH, (int)padW);
        if(WITH_BIAS) {
            channelReduce(Y->dA, bias->grad(), reduceBuf);
        }
    }
    
    void Conv2D::updateParams() {
        filter->opt->apply();
        if(WITH_BIAS){
            bias->opt->apply();
        }
    }
    
    void Conv2D::batchUpdateParams() {
        filter->opt->batchApply();
        if(WITH_BIAS){
            bias->opt->batchApply();
        }
    }
    
    void Conv2D::randFillNetParams() {
        uint32 K = filter->data()->dims.size / filter->data()->dims.n;
        filter->data()->randNormal(0, (float)sqrt(2.0 / (float) K));
        if (WITH_BIAS)
            bias->data()->randNormal(0, (float)sqrt(2.0 / (float)filter->data()->dims.n));
    }
    
    void Conv2D::zeroGrads() {
        filter->opt->zeroGrad();
        if(WITH_BIAS){
            bias->opt->zeroGrad();
        }
    }
    
    uint32 Conv2D::encodeInfo(fstream *fout, uint64 offset) {
        uint32 runningOffset = offset;
        uint32 n = WITH_BIAS ? 1 : 0;
        runningOffset += filter->encodeNetParamInfo(fout, runningOffset);
        fout->seekp(runningOffset);
        fout->write((char*)&strideH, sizeof(uint32));
        fout->write((char*)&strideW, sizeof(uint32));
        fout->write((char*)&padH, sizeof(uint32));
        fout->write((char*)&padW, sizeof(uint32));
        fout->write((char*)&n, sizeof(uint32));
        runningOffset += sizeof(uint32) * 5;
        return runningOffset - offset;
    }
    
    uint32 Conv2D::encodeNetParams(fstream *fout, uint64 offset) {
        uint32 runningOffset = offset;
        runningOffset += filter->encodeNetParamData(fout, runningOffset);
        if(WITH_BIAS){
            runningOffset += bias->encodeNetParamData(fout, runningOffset);
        }
        return runningOffset - offset;
    }
} // seann