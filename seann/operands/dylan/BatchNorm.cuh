//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_BATCHNORM_CUH
#define CRUSADER_BATCHNORM_CUH

#include "../OperandBase.cuh"

namespace seann {
    class BatchNorm : public OperandBase {
    public:
        NetParam* beta{};
        NetParam* gamma{};
        Tensor* mean{};
        Tensor* variance{};
        
        BatchNorm() : OperandBase() {}
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override {
            shape4 paramShape = {1, inShape.c, inShape.h, inShape.w};
            beta = new NetParam(info, paramShape);
            gamma = new NetParam(info, paramShape);
            
            mean = Tensor::create(paramShape);
            variance = Tensor::create(paramShape);
            
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
    };
    
} // seann

#endif //CRUSADER_BATCHNORM_CUH
