//
// Created by Dylan on 7/10/2022.
//

#ifndef CRUSADER_AVGPOOL2D_CUH
#define CRUSADER_AVGPOOL2D_CUH

#include "../../OperandBase.cuh"

#define OPR_SEBLAS_AVGPOOL2D 0xb001

namespace seann {
    
    OperandBase* DEC_OPR_AVGPOOL2D_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_AVGPOOL2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    class AvgPool2D : public OperandBase {
    public:
        uint32 strideH;
        uint32 strideW;
        uint32 rangeH;
        uint32 rangeW;
    
        AvgPool2D(uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
            this->strideH = strideH;
            this->strideW = strideW;
            this->rangeH = rangeH;
            this->rangeW = rangeW;
            
            decodeInfo = DEC_OPR_AVGPOOL2D_INFO;
            decodeParams = DEC_OPR_AVGPOOL2D_PARAM;
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
            Y = Parameter::create(inShape.n, inShape.c, inShape.h / strideH, inShape.w / strideW);
        }
    
        void zeroGrads() override{}
    
        uint32 OPERAND_ID() override {
            return OPR_SEBLAS_AVGPOOL2D;
        }
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
    };
    
} // seann

#endif //CRUSADER_AVGPOOL2D_CUH
