//
// Created by Dylan on 6/27/2022.
//

#ifndef CRUSADER_DATATRANSFORM_CUH
#define CRUSADER_DATATRANSFORM_CUH

#include "../../seblas/tensor/Tensor.cuh"

using namespace seblas;

namespace seio {
    class DataTransformer{
    public:
        virtual Tensor* apply(Tensor* X) = 0;
    };
    
    class DistribNorm : public DataTransformer{
    public:
        float desiredMean = 0;
        float desiredVar = 1;
        
        DistribNorm(float mean, float var){
            desiredMean = mean;
            desiredVar = var;
        }
        
        DistribNorm() = default;
        
        Tensor* apply(Tensor* X) override;
    };
    
    class UniformNorm : public DataTransformer{
    public:
        float desiredMin = 0;
        float desiredMax = 1;
        
        UniformNorm(float min, float max){
            desiredMin = min;
            desiredMax = max;
        }
        
        UniformNorm() = default;
        
        Tensor* apply(Tensor* X) override;
    };
}


#endif //CRUSADER_DATATRANSFORM_CUH
