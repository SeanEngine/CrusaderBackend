//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_LINEAR_CUH
#define CRUSADER_LINEAR_CUH

#include "../OperandBase.cuh"

#define OPR_SEBLAS_LINEAR 0x0001

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
        
        uint32 OPERAND_ID() override {
            return OPR_SEBLAS_LINEAR;
        }
        
        float getOptimLR() override;
        
        void updateOptimLR(float val) override;
        
        float getL2Const() override{
            return weights->opt->L2;
        }
        
        void updateL2Const(float val) override{
            weights->opt->L2 = val;
            biases->opt->L2 = val;
        }
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
    };
    
} // seann

#endif //CRUSADER_LINEAR_CUH
