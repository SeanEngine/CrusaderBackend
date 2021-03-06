//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_MAXPOOL2D_CUH
#define CRUSADER_MAXPOOL2D_CUH

#include "../../OperandBase.cuh"

#define OPR_SEBLAS_MAXPOOL2D 0xb003

namespace seann {
    
    class MaxPool2D : public OperandBase{
    public:
        Tensor* record{};
        uint32 strideH;
        uint32 strideW;
        uint32 rangeH;
        uint32 rangeW;
    
        MaxPool2D(uint32 strideH, uint32 strideW, uint32 rangeH, uint32 rangeW){
            this->strideH = strideH;
            this->strideW = strideW;
            this->rangeH = rangeH;
            this->rangeW = rangeW;
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
            record = Tensor::create(
                inShape.n, inShape.c,
                ((inShape.h - (rangeH - strideH)) / strideH) * rangeH,
                ((inShape.w - (rangeW - strideW)) / strideW) * rangeW
            );
        }
        
        void zeroGrads() override{}
        
        uint32 OPERAND_ID() override {
            return OPR_SEBLAS_MAXPOOL2D;
        }
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
        
        uint32 getInfoEncodingSize() override {
            return sizeof(uint32) * 4;
        }
        
        uint32 getNetParamsEncodingSize() override {
            return 0;
        }
    };
    
} // seamm

#endif //CRUSADER_MAXPOOL2D_CUH
