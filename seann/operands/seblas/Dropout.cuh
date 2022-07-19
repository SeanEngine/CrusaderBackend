//
// Created by Dylan on 6/23/2022.
//

#ifndef CRUSADER_DROPOUT_CUH
#define CRUSADER_DROPOUT_CUH

#include "../OperandBase.cuh"
#define OPR_SEBLAS_DROPOUT 0x0004


namespace seann {
    
    class Dropout : public OperandBase{
    public:
        float p;
        Tensor* mask;
        
        explicit Dropout(float prop) : OperandBase() {
            p = prop;
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
        
        uint32 getInfoEncodingSize() override {
            return sizeof(float);
        }
        
        uint32 getNetParamsEncodingSize() override {
            return 0;
        }
    };
} // seann

#endif //CRUSADER_DROPOUT_CUH
