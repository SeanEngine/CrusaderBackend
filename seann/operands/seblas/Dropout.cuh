//
// Created by Dylan on 6/23/2022.
//

#ifndef CRUSADER_DROPOUT_CUH
#define CRUSADER_DROPOUT_CUH

#include "../OperandBase.cuh"
#define OPR_SEBLAS_DROPOUT 0x0004


namespace seann {
    
    OperandBase* DEC_OPR_DROPOUT_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_DROPOUT_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    class Dropout : public OperandBase{
    public:
        float p;
        Tensor* mask;
        
        explicit Dropout(float prop) : OperandBase() {
            p = prop;
            decodeInfo = DEC_OPR_DROPOUT_INFO;
            decodeParams = DEC_OPR_DROPOUT_PARAM;
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
            return OPR_SEBLAS_DROPOUT;
        }
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
    };
} // seann

#endif //CRUSADER_DROPOUT_CUH
