//
// Created by Dylan on 6/23/2022.
//

#ifndef CRUSADER_DROPOUT_CUH
#define CRUSADER_DROPOUT_CUH

#include "../OperandBase.cuh"

namespace seann {
    
    class Dropout : public OperandBase{
    public:
        float p;
        Tensor* mask;
        
        explicit Dropout(float prop) : OperandBase() {
            p = prop;
        }
        
        void forward() override;
        
        void xGrads() override;
        
        void paramGrads() override{}
        
        void zeroGrads() override{}
        
        string info() override;
        
        void batchUpdateParams() override{}
        
        void updateParams() override{}
        
        void randFillNetParams() override{}
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override;
        
        void inferenceForward() override;
        
        uint32 OPERAND_ID() override {
            return 0x0b02;
        }
    };
} // seann

#endif //CRUSADER_DROPOUT_CUH
