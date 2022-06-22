//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_MAXPOOL2D_CUH
#define CRUSADER_MAXPOOL2D_CUH

#include "OperandBase.cuh"

namespace seann {
    class MaxPool2D : public OperandBase{
    public:
        Tensor* record{};
        uint32 stepH;
        uint32 stepW;
        explicit MaxPool2D(uint32 stepH, uint32 stepW){
            this->stepH = stepH;
            this->stepW = stepW;
        }
        
        void randFillNetParams() override{}
        
        string info() override;
        
        void paramGrads() override{}
        
        void xGrads() override;
        
        void forward() override;
        
        void updateParams() override{}
        
        void batchUpdateParams() override{}
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape.n, inShape.c, inShape.h / stepH, inShape.w / stepW);
            record = Tensor::create(inShape);
        }
        
        void zeroGrads() override{}
    };
    
} // seamm

#endif //CRUSADER_MAXPOOL2D_CUH
