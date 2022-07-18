//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_SOFTMAX_CUH
#define CRUSADER_SOFTMAX_CUH

#include "../OperandBase.cuh"
#define OPR_SEBLAS_SOFTMAX 0x0005

namespace seann {
    
    OperandBase* DEC_OPR_SOFTMAX_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_SOFTMAX_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    class Softmax : public OperandBase {
    public:
        uint32 INPUT_SIZE{};
        Tensor* reduceBuffer{};
        Softmax(){}
        
        string info() override {
            return "Softmax       { " + to_string(INPUT_SIZE) + " }";
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            INPUT_SIZE = inShape.size / inShape.n;
            X = Parameter::declare(inShape.n, 1, INPUT_SIZE, 1);
            Y = Parameter::create(inShape.n, 1, INPUT_SIZE, 1);
            reduceBuffer = INPUT_SIZE / 1024 > 0 ? Tensor::create(INPUT_SIZE,1) : nullptr;
            
            decodeInfo = DEC_OPR_SOFTMAX_INFO;
            decodeParams = DEC_OPR_SOFTMAX_PARAM;
        }
        
        void forward() override;
        
        void xGrads() override;
        
        void batchUpdateParams() override{}
        
        void updateParams() override{}
        
        void paramGrads() override{}
        
        void randFillNetParams() override{}
        
        void zeroGrads() override{}
        
        uint32 OPERAND_ID() override {
            return OPR_SEBLAS_SOFTMAX;
        }
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
    };
} // seann

#endif //CRUSADER_SOFTMAX_CUH
