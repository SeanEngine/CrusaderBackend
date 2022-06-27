//
// Created by Dylan on 6/27/2022.
//

#ifndef CRUSADER_DATA_CUH
#define CRUSADER_DATA_CUH

#include "../../seblas/tensor/Tensor.cuh"

using namespace seblas;

namespace seio {
    struct Data{
    public:
        Tensor* X;
        Tensor* label;
        
        static Data* declare(shape4 dataShape, shape4 labelShape);
        
        Data* instantiate();
        
        Data* instantiateHost();
        
        void destroy();
        
        void destroyHost();
        
        Data* inherit(Tensor* X0, Tensor* label0);
        
        Data* copyOffD2D(Data* onDevice);
        
        Data* copyOffH2D(Data* onHost);
        
        Data* copyOffD2H(Data* onDevice);
        
        Data* copyOffH2H(Data* onHost);
        
        Data* copyOffD2D(Tensor* X0, Tensor* label0);
        
        Data* copyOffH2D(Tensor* X0, Tensor* label0);
        
        Data* copyOffD2H(Tensor* X0, Tensor* label0);
        
        Data* copyOffH2H(Tensor* X0, Tensor* label0);
    };
    
    
} // seio

#endif //CRUSADER_DATA_CUH
