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
        convDerive(filter->data(), Y->dA, X->dA, (int)strideH, (int)strideW, (int)padH, (int)padW);
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
} // seann