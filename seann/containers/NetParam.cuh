//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_NETPARAM_CUH
#define CRUSADER_NETPARAM_CUH

#include "../optimizers/Optimizers.cuh"


namespace seann {
    /**
     * Netparams are parameters that are actually upgrading
     * such as weights and biases.
     * each net param will contain a optimizer
     */
    class NetParam {
    public:
        Parameter* A;
        Optimizer* opt;
        
        NetParam(Parameter* A, Optimizer* opt) : A(A), opt(opt) {}
        
        NetParam(Parameter* A, OptimizerInfo* info) : A(A) {
            opt = info->create(A);
        }
        
        template<typename... Args>
        explicit NetParam(OptimizerInfo* info, Args&&... args) {
            A = Parameter::create(std::forward<Args>(args)...);
            opt = info->create(A);
        }
        
        void setWeightDecay(float weightDecay) const {
            opt->L2 = weightDecay;
        }
        
        void update() const {
            opt->apply();
        }
        
        NetParam* setWeight() {
            opt->isWeight = true;
            return this;
        }
        
        Tensor* data() const{
            return A->A;
        }
        
        Tensor* grad() const{
            return A->dA;
        }
    };
    
} // seann

#endif //CRUSADER_NETPARAM_CUH
