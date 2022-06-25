//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_CONV2D_CUH
#define CRUSADER_CONV2D_CUH

#include "../OperandBase.cuh"

namespace seann {
    class Conv2D : public OperandBase {
    public:
        shape4 filterShape;
        NetParam* filter{};
        NetParam* bias = nullptr;
        uint32 strideH;
        uint32 strideW;
        uint32 padH;
        uint32 padW;
        
        bool WITH_BIAS = false;
        Tensor* reduceBuf{}; //for calculating gradients of bias
        
        Conv2D(shape4 filterShape, uint32 strideH, uint32 strideW, uint32 padH, uint32 padW, bool WITH_BIAS)
                : filterShape(filterShape), strideH(strideH), strideW(strideW), padH(padH), padW(padW) {
            this->WITH_BIAS = WITH_BIAS;
        }
        
        string info() override {
            return "Conv2D        { filter: " + filter->data()->dims.toString() + ", input feature: " + X->A->dims.toString() + " }";
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override {
            filter = new NetParam(info, filterShape);
            if (WITH_BIAS) bias = new NetParam(info, filterShape.n, 1);
            X = Parameter::declare(inShape); //input features
            shape4 outShape = {
                    X->A->dims.n,
                    filterShape.n,
                    (inShape.h + 2 * padH - filterShape.h) / strideH + 1,
                    (inShape.w + 2 * padW - filterShape.w) / strideW + 1};
            
            Y = Parameter::create(outShape);
            if(WITH_BIAS) {
                reduceBuf = outShape.h * outShape.w > 1024 ?
                            Tensor::declare(filterShape.n, outShape.h * outShape.w / 1024) : nullptr;
            }
        }
        
        void forward() override;
        
        void xGrads() override;
        
        void paramGrads() override;
        
        void updateParams() override;
        
        void batchUpdateParams() override;
        
        void randFillNetParams() override;
        
        void zeroGrads() override;
    };
    
} // seann

#endif //CRUSADER_CONV2D_CUH
