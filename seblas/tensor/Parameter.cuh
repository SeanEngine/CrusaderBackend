//
// Created by Dylan on 6/17/2022.
//

#ifndef CRUSADER_PARAMETER_CUH
#define CRUSADER_PARAMETER_CUH

#include "Tensor.cuh"

namespace seblas {
    class Parameter {
    public:
        Tensor* A;
        Tensor* dA;
        
        static Parameter* create(shape4 dims){
            Parameter* p;
            cudaMallocHost(&p, sizeof(Parameter));
            p->A = Tensor::declare(dims)->instantiate();
            p->dA = Tensor::declare(dims)->instantiate();
            return p;
        };
        
        template<typename... Args>
        static Parameter* create(Args &&... args) {
            return create(shape4(std::forward<Args>(args)...));
        }
    
        static Parameter* create(Tensor* src){
            Parameter* p;
            cudaMallocHost(&p, sizeof(Parameter));
            p->A = src;
            p->dA = Tensor::create(src->dims);
            return p;
        };
        
        static Parameter* declare(shape4 dims){
            Parameter* p;
            cudaMallocHost(&p, sizeof(Parameter));
            p->A = Tensor::declare(dims);
            p->dA = Tensor::declare(dims);
            return p;
        }
    
        template<typename... Args>
        static Parameter* declare(Args &&... args) {
            return declare(shape4(std::forward<Args>(args)...));
        }
        
        
        Parameter* instantiate(){
            A->instantiate();
            dA->instantiate();
            return this;
        }
        
        //inherit keeps the shape of the parameter
        //but uses the same element memory block as another parameter
        Parameter* inherit(Parameter* src){
            assert(src->A->dims.size == this->A->dims.size);
            this->A->attach(src->A);
            this->dA->attach(src->dA);
            return this;
        }
    };
    
} // seblas

#endif //CRUSADER_PARAMETER_CUH
