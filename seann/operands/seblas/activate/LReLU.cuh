//
// Created by Dylan on 7/16/2022.
//

#ifndef CRUSADER_LRELU_CUH
#define CRUSADER_LRELU_CUH

#include "../../OperandBase.cuh"

namespace seann {
    
    class LReLU : public OperandBase{
    public:
        uint32 INPUT_SIZE{};
        uint32 PARALELL_SIZE{};
        float alpha;
        explicit LReLU(float alpha){
            this->alpha = alpha;
        }
    
        string info() override {
            return "LReLU         { " + std::to_string(INPUT_SIZE/PARALELL_SIZE) + " }";
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

#endif //CRUSADER_LRELU_CUH
