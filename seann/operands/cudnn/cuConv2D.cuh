//
// Created by Dylan on 6/25/2022.
//

#ifndef CRUSADER_CUCONV2D_CUH
#define CRUSADER_CUCONV2D_CUH

#include "../OperandBase.cuh"

namespace seann {
    
    class cuConv2D : public OperandBase {
    public:
        cudnnHandle_t cudnn;
        cudnnFilterDescriptor_t filterDesc{};
        cudnnConvolutionDescriptor_t convDesc{};
    
        shape4 filterShape;
        NetParam* filter{};
        NetParam* bias = nullptr;
    
        uint32 strideH;
        uint32 strideW;
        uint32 padH;
        uint32 padW;
        uint32 dilationH = 1;
        uint32 dilationW = 1;
    
        bool WITH_BIAS = false;
    
        cuConv2D(cudnnHandle_t cudnn, shape4 filterShape, uint32 strideH, uint32 strideW, uint32 padH, uint32 padW, bool WITH_BIAS)
        : filterShape(filterShape), strideH(strideH), strideW(strideW), padH(padH), padW(padW) {
            this->WITH_BIAS = WITH_BIAS;
    
            cudnnCreateConvolutionDescriptor(&convDesc);
            cudnnCreateFilterDescriptor(&filterDesc);
            
            cudnnSetConvolution2dDescriptor(convDesc,
                                            (int)padH,
                                            (int)padW,
                                            (int)strideH,
                                            (int)strideW,
                                            (int)dilationH,
                                            (int)dilationW,
                                            CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT);
            
            cudnnSetFilter4dDescriptor(filterDesc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                       (int)filterShape.n,
                                       (int)filterShape.c,
                                       (int)filterShape.h,
                                       (int)filterShape.w);
            
            this->cudnn = cudnn;
        }
        
        string info() override {
            return "cuConv2D        { filter: " + filterShape.toString() + ","
                    + " input feature: " + X->A->dims.toString() + " }";
        }
    
        void initNetParams(OptimizerInfo *info, shape4 inShape) override;
    
        void forward() override;
    
        void xGrads() override;
    
        void paramGrads() override;
    
        void updateParams() override;
    
        void batchUpdateParams() override;
    
        void randFillNetParams() override;
    
        void zeroGrads() override;
        
        uint32 OPERAND_ID() override {
            return 0x0ad2;
        }
        
        float getOptimLR() override {
            return filter->opt->LEARNING_RATE;
        }
        
        void updateOptimLR(float val) override {
            filter->opt->LEARNING_RATE = val;
            if (WITH_BIAS) bias->opt->LEARNING_RATE = val;
        }
        
        float getL2Const() override {
            return filter->opt->L2;
        }
        
        void updateL2Const(float val) override {
            filter->opt->L2 = val;
            if (WITH_BIAS) bias->opt->L2 = val;
        }
    };
    
} // seann

#endif //CRUSADER_CUCONV2D_CUH
