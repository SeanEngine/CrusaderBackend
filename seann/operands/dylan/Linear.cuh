//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_LINEAR_CUH
#define CRUSADER_LINEAR_CUH

#include "../OperandBase.cuh"

namespace seann {
    class Linear : public OperandBase {
    public:
        NetParam* weights{};
        NetParam* biases{};
        uint32 INPUT_SIZE{};
        uint32 OUTPUT_SIZE;
        
        explicit Linear(uint32 OUTPUT_SIZE){
            this->OUTPUT_SIZE = OUTPUT_SIZE;
        }
        
        string info() override {
            return "Linear        { " + to_string(INPUT_SIZE) + ", " + to_string(OUTPUT_SIZE) + " }";
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override;
        
        void forward() override;
        
        void paramGrads() override;
        
        void updateParams() override;
        
        void batchUpdateParams() override;
        
        void xGrads() override;
        
        void randFillNetParams() override;
        
        void zeroGrads() override;
    };
    
} // seann

#endif //CRUSADER_LINEAR_CUH
