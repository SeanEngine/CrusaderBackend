//
// Created by Dylan on 6/19/2022.
//

#ifndef CRUSADER_OPTIMIZERS_CUH
#define CRUSADER_OPTIMIZERS_CUH

#include "../../seblas/operations/cuOperations.cuh"

using namespace seblas;
namespace seann {
    
    /**
     * @brief for all the optimisers, exists the following functions:
     *
     * parameter to optimize : θ, cost function : C(θ), LEARNING_RATE : η
     *
     * 1. calculate the gradients of parameters to update: gt = ▽C(θt)
     * 2. calculate 1st and 2nd order momentum: m = Φ(g1,g2,...,gt), v = φ(g1,g2,...,gt)
     * 3. calculate update amount: θt' = η * m / (sqrt(v) + ε)
     * 4. update parameters: θt+1 = θt - θt'
     */
    struct Optimizer {
    public:
        float LEARNING_RATE{};
        Parameter* A;
        
        explicit Optimizer(float LEARNING_RATE, Parameter* A)
                : LEARNING_RATE(LEARNING_RATE), A(A){}
        
        
        //apply the gradient to the parameters (weights, biases, etc)
        virtual void apply() = 0;
        
        //updates relative to batches
        virtual void batchApply() = 0;
        
        virtual void zeroGrad() = 0;
    };
    
    
    //SGD optimizer : Stochastic Gradient Decent, w = w - η * g
    //This optimizer does not have batch behaviors
    struct SGD : public Optimizer {
    public:
        explicit SGD(float LEARNING_RATE, Parameter* A)
                : Optimizer(LEARNING_RATE, A){}
        
        void apply() override;
        void batchApply() override{}
        void zeroGrad() override;
    };
    
    
    //BGD optimizer : Batch Gradient Decent, w = w - (η * sum_batch(g))/bs
    struct BGD : public Optimizer {
    public:
        float BATCH_SIZE;
        explicit BGD(float LEARNING_RATE, Parameter* A, float BATCH_SIZE)
                : Optimizer(LEARNING_RATE, A), BATCH_SIZE(BATCH_SIZE){}
        
        void apply() override {}
        
        void batchApply() override;
        
        void zeroGrad() override;
    };
    
    
    //Momentum : m[t] = m[t-1] * β + (1 - β) * g[t]
    //           w[t] = w[t-1] - η * m[t]
    struct Momentum : public Optimizer {
    public:
        float BETA = 0.9;
        Tensor* m;
        
        explicit Momentum(float LEARNING_RATE, Parameter* A)
                : Optimizer(LEARNING_RATE, A){
            m = Tensor::create(A->dA->dims);
        }
        
        explicit Momentum(float LEARNING_RATE, Parameter* A, float BETA)
                : Momentum(LEARNING_RATE, A){
            this->BETA = BETA;
        }
        
        void apply() override;
        void batchApply() override{}
        void zeroGrad() override;
    };
    
    
    //AdaGrad : V[t] = sumOf(g[1]^2 ... g[t]^2)
    //          w[t] = w[t-1] - η * g[t] / (sqrt(V[t]) + ε)
    struct AdaGrad : public Optimizer {
    public:
        float EPSILON = 1e-10;
        Tensor* V;
        
        explicit AdaGrad(float LEARNING_RATE, Parameter* A)
                : Optimizer(LEARNING_RATE, A){
            V = Tensor::create(A->dA->dims)->constFill(0);
        }
        
        explicit AdaGrad(float LEARNING_RATE, Parameter* A, float EPSILON)
                : AdaGrad(LEARNING_RATE, A){
            this->EPSILON = EPSILON;
        }
        
        void apply() override;
        void batchApply() override{}
        void zeroGrad() override;
    };
    
    /*
     * Compute Gradient: gt
     * Accumulate Gradient: E[g^2]t = ρE[g^2]t−1 + (1 − ρ)g^2t
     * Compute Update: ∆xt = −(RMS[∆x]t−1/RMS[g]t) * gt
     * Accumulate Updates: E[∆x^2]t = ρE[∆x^2]t−1+(1−ρ)∆x^2t
     * Update Parameters: x[t+1] = x[t] + ∆xt
     *
     * RMS : (E[g^2]t + ε)^(1/2)
     *
     * Check Paper: https://arxiv.org/abs/1212.5701
     */
    struct AdaDelta : public Optimizer {
    public:
        float BETA = 0.95;
        float EPSILON = 1e-10;
        Tensor* V;
        Tensor* Vx;
        
        explicit AdaDelta(float LEARNING_RATE, Parameter* A, float BETA, float EPSILON)
                : Optimizer(LEARNING_RATE, A), BETA(BETA), EPSILON(EPSILON){
            V = Tensor::create(A->dA->dims)->constFill(0);
            Vx = Tensor::create(A->dA->dims)->constFill(0);
        }
        
        void apply() override;
        
        void batchApply() override{}
        
        void zeroGrad() override;
    };
    
    //Adaptive Momentum : m[t] = m[t-1] * β1 + (1 - β1) * g[t]
    //                    V[t] = V[t-1] * β2 + (1 - β2) * g[t]^2
    //                    w[t] = w[t-1] - η * m[t] / (sqrt(V[t]) + ε)
    struct Adam : public Optimizer {
    public:
        float BETA1 = 0.9;
        float BETA2 = 0.999;
        float EPSILON = 1e-10;
        uint32 t = 0;
        Tensor* m;
        Tensor* V;
        
        explicit Adam(float LEARNING_RATE, Parameter* A)
                : Optimizer(LEARNING_RATE, A){
            m = Tensor::create(A->dA->dims);
            V = Tensor::create(A->dA->dims);
        }
        
        explicit Adam(float LEARNING_RATE, Parameter* A, float BETA1, float BETA2)
                : Optimizer(LEARNING_RATE, A), BETA1(BETA1), BETA2(BETA2){
            m = Tensor::create(A->dA->dims);
            V = Tensor::create(A->dA->dims);
        }
        
        explicit Adam(float LEARNING_RATE, Parameter* A, float BETA1, float BETA2, float EPSILON)
                : Optimizer(LEARNING_RATE, A), EPSILON(EPSILON), BETA1(BETA1), BETA2(BETA2){
            m = Tensor::create(A->dA->dims);
            V = Tensor::create(A->dA->dims);
        }
        
        void apply() override;
        
        void batchApply() override{}
        
        void zeroGrad() override;
    };
    
    //these are the templates used to instantiate optimizer for each updating parameter
    //A template for the model will allow all operands initialize optimizers based on it
    struct OptimizerInfo {
        virtual Optimizer* create(Parameter* A) = 0;
    };
    
    struct OPTIMIZER_SGD : public OptimizerInfo {
        float LEARNING_RATE;
        explicit OPTIMIZER_SGD(float LEARNING_RATE) : LEARNING_RATE(LEARNING_RATE){}
        
        Optimizer* create(Parameter* A) override {
            return new SGD(LEARNING_RATE, A);
        }
    };
    
    struct OPTIMIZER_MOMENTUM : public OptimizerInfo {
        float LEARNING_RATE;
        float BETA = 0.9;
        explicit OPTIMIZER_MOMENTUM(float LEARNING_RATE, float BETA) : LEARNING_RATE(LEARNING_RATE), BETA(BETA){}
        explicit OPTIMIZER_MOMENTUM(float LEARNING_RATE) : LEARNING_RATE(LEARNING_RATE){}
        
        Optimizer* create(Parameter* A) override {
            return new Momentum(LEARNING_RATE, A, BETA);
        }
    };
    
    struct OPTIMIZER_ADAGRAD : public OptimizerInfo {
        float LEARNING_RATE;
        float EPSILON = 1e-10;
        explicit OPTIMIZER_ADAGRAD(float LEARNING_RATE, float EPSILON) : LEARNING_RATE(LEARNING_RATE), EPSILON(EPSILON){}
        explicit OPTIMIZER_ADAGRAD(float LEARNING_RATE) : LEARNING_RATE(LEARNING_RATE){}
        
        Optimizer* create(Parameter* A) override {
            return new AdaGrad(LEARNING_RATE, A, EPSILON);
        }
    };
    
    struct OPTIMIZER_ADADELTA : public OptimizerInfo {
        float LEARNING_RATE;
        float BETA = 0.99;
        float EPSILON = 1e-10;
        explicit OPTIMIZER_ADADELTA(float LEARNING_RATE, float BETA, float EPSILON) :
                LEARNING_RATE(LEARNING_RATE), BETA(BETA), EPSILON(EPSILON){}
        explicit OPTIMIZER_ADADELTA(float LEARNING_RATE, float BETA) :
                LEARNING_RATE(LEARNING_RATE), BETA(BETA){}
        explicit OPTIMIZER_ADADELTA(float LEARNING_RATE) : LEARNING_RATE(LEARNING_RATE){}
        
        Optimizer* create(Parameter* A) override {
            return new AdaDelta(LEARNING_RATE, A, BETA, EPSILON);
        }
    };
    
    struct OPTIMIZER_ADAM : public OptimizerInfo {
        float LEARNING_RATE;
        float BETA1 = 0.9;
        float BETA2 = 0.99;
        float EPSILON = 1e-10;
        explicit OPTIMIZER_ADAM(float LEARNING_RATE, float BETA1, float BETA2, float EPSILON) :
                LEARNING_RATE(LEARNING_RATE), BETA1(BETA1), BETA2(BETA2), EPSILON(EPSILON){}
        explicit OPTIMIZER_ADAM(float LEARNING_RATE, float BETA1, float BETA2) :
                LEARNING_RATE(LEARNING_RATE), BETA1(BETA1), BETA2(BETA2){}
        explicit OPTIMIZER_ADAM(float LEARNING_RATE) : LEARNING_RATE(LEARNING_RATE){}
        
        Optimizer* create(Parameter* A) override {
            return new Adam(LEARNING_RATE, A, BETA1, BETA2, EPSILON);
        }
    };
    
} // seann

#endif //CRUSADER_OPTIMIZERS_CUH
