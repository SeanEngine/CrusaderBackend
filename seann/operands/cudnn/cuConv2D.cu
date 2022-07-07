//
// Created by Dylan on 6/25/2022.
//

#include "cuConv2D.cuh"

namespace seann {
    void cuConv2D::initNetParams(OptimizerInfo *info, shape4 inShape) {
        filter = (new NetParam(info, filterShape))->setWeight();
        if (WITH_BIAS) bias = new NetParam(info, filterShape.n, 1);
        X = Parameter::declare(inShape); //input features
        shape4 outShape = {
                X->A->dims.n,
                filterShape.n,
                (inShape.h + 2 * padH - filterShape.h) / strideH + 1,
                (inShape.w + 2 * padW - filterShape.w) / strideW + 1};
    
        Y = Parameter::create(outShape);
    }
    
    void cuConv2D::forward() {
        float alpha = 1.0f, beta = 0.0f;
        cudnnConvolutionForward(
                cudnn,
                &alpha,
                X->A->cudnnDesc, X->A->elements,
                filterDesc, filter->data()->elements,
                 convDesc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                nullptr, 0,
                &beta,
                Y->A->cudnnDesc, Y->A->elements);
    }
    
    void cuConv2D::xGrads() {
        float alpha = 1.0f, beta = 0.0f;
        *Y->dA + Y->dAReserve;
        cudnnConvolutionBackwardData(
                cudnn,
                &alpha,
                filterDesc, filter->data()->elements,
                Y->dA->cudnnDesc, Y->dA->elements,
                convDesc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                nullptr, 0,
                &beta,
                X->dA->cudnnDesc, X->dA->elements);
        Y->dAReserve->constFill(0);
    }
    
    void cuConv2D::paramGrads() {
        float alpha = 1.0f, beta = 0.0f;
        cudnnConvolutionBackwardFilter(
                cudnn,
                &alpha,
                X->A->cudnnDesc, X->A->elements,
                Y->dA->cudnnDesc, Y->dA->elements,
                convDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                nullptr, 0,
                &beta,
                filterDesc, filter->grad()->elements);
        
        if(WITH_BIAS){
            cudnnConvolutionBackwardBias(
                    cudnn,
                    &alpha,
                    Y->dA->cudnnDesc, Y->dA->elements,
                    &beta,
                    bias->grad()->cudnnDesc, bias->grad()->elements
                    );
        }
    }
    
    void cuConv2D::updateParams() {
        filter->opt->apply();
        if(WITH_BIAS) bias->opt->apply();
    }
    
    void cuConv2D::batchUpdateParams() {
        filter->opt->batchApply();
        if(WITH_BIAS) bias->opt->batchApply();
    }
    
    void cuConv2D::randFillNetParams() {
        uint32 K = filter->data()->dims.size / filter->data()->dims.n;
        filter->data()->randNormal(0, (float)sqrt(2.0 / (float) K));
        if (WITH_BIAS)
            bias->data()->randNormal(0, (float)sqrt(2.0 / (float)filter->data()->dims.n));
    }
    
    void cuConv2D::zeroGrads() {
        filter->opt->zeroGrad();
        if (WITH_BIAS) bias->opt->zeroGrad();
    }
    
} // seann