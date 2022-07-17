//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_RELU_CUH
#define CRUSADER_RELU_CUH

#include "../../OperandBase.cuh"

namespace seann {
    class ReLU : public OperandBase {
    public:
        uint32 INPUT_SIZE{};
        uint32 PARALELL_SIZE{};
        ReLU() = default;
        
        string info() override {
            return "ReLU          { " + std::to_string(INPUT_SIZE/PARALELL_SIZE) + " }";
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            INPUT_SIZE = inShape.size;
            PARALELL_SIZE = inShape.n;
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape);
        }
        
        void forward() override;
        
        void xGrads() override;
        
        void batchUpdateParams() override{}
        
        void updateParams() override{}
        
        void paramGrads() override{}
        
        void randFillNetParams() override{}
        
        void zeroGrads() override{}
        
        uint32 OPERAND_ID() override {
            return 0x0c01;
        }
    };
    
} // seann

#endif //CRUSADER_RELU_CUH
