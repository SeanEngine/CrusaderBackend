//
// Created by Dylan on 7/16/2022.
//

#ifndef CRUSADER_GLOBALAVGPOOL_CUH
#define CRUSADER_GLOBALAVGPOOL_CUH

#include "../../OperandBase.cuh"

namespace seann {
    
    class GlobalAvgPool : public OperandBase {
    public:
        Tensor* buffer = nullptr;  //for reduction
        
        GlobalAvgPool() = default;
    
        void randFillNetParams() override{}
    
        string info() override;
    
        void paramGrads() override{}
    
        void xGrads() override;
    
        void forward() override;
    
        void updateParams() override{}
    
        void batchUpdateParams() override{}
    
        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape.n, inShape.c, 1, 1);
            if(X->A->dims.h * X->A->dims.w > REDUCE_BLOCK){
                buffer = Tensor::create(inShape);
            }
        }
    
        void zeroGrads() override{}
    
        uint32 OPERAND_ID() override {
            return 0x0b04;
        }
    };
    
} // seann

#endif //CRUSADER_GLOBALAVGPOOL_CUH
