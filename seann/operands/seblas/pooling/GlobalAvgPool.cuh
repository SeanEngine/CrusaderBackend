//
// Created by Dylan on 7/16/2022.
//

#ifndef CRUSADER_GLOBALAVGPOOL_CUH
#define CRUSADER_GLOBALAVGPOOL_CUH

#include "../../OperandBase.cuh"

#define OPR_SEBLAS_GLOBALAVGPOOL 0xb002

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
            return OPR_SEBLAS_GLOBALAVGPOOL;
        }
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
    };
    
} // seann

#endif //CRUSADER_GLOBALAVGPOOL_CUH
