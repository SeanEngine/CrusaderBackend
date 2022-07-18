//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_BATCHNORM_CUH
#define CRUSADER_BATCHNORM_CUH

#include "../OperandBase.cuh"

#define OPR_SEBLAS_BATCHNORM 0x0003

//TODO: change this BN to allow mean and var cumulation and proper inference

namespace seann {
    
    OperandBase* DEC_OPR_BATCHNORM_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_BATCHNORM_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    class BatchNorm : public OperandBase {
    public:
        NetParam* beta{};
        NetParam* gamma{};
        Tensor* mean{};
        Tensor* variance{};
        Tensor* xHatP{};
        
        BatchNorm() : OperandBase() {
            decodeInfo = DEC_OPR_BATCHNORM_INFO;
            decodeParams = DEC_OPR_BATCHNORM_PARAM;
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override {
            shape4 paramShape = {1, inShape.c, inShape.h, inShape.w};
            beta = new NetParam(info, paramShape);
            gamma = new NetParam(info, paramShape);
            
            mean = Tensor::create(paramShape);
            variance = Tensor::create(paramShape);
            xHatP = Tensor::create(inShape);
            
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape);
        }
        
        void randFillNetParams() override;
        
        void forward() override;
        
        void xGrads() override;
        
        void paramGrads() override;
        
        void updateParams() override;
        
        void batchUpdateParams() override;
        
        void zeroGrads() override;
        
        string info() override;
        
        uint32 OPERAND_ID() override {
            return OPR_SEBLAS_BATCHNORM;
        }
        
        float getOptimLR() override {
            return gamma->opt->LEARNING_RATE;
        }
        
        void updateOptimLR(float lr) override {
            gamma->opt->LEARNING_RATE = lr;
            beta->opt->LEARNING_RATE = lr;
        }
        
        float getL2Const() override {
            return gamma->opt->L2;
        }
        
        void updateL2Const(float l2) override {
            gamma->opt->L2 = l2;
            beta->opt->L2 = l2;
        }
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
    };
    
} // seann

#endif //CRUSADER_BATCHNORM_CUH
