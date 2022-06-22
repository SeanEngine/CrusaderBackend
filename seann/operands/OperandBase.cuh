//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_OPERANDBASE_CUH
#define CRUSADER_OPERANDBASE_CUH

#include "../../seblas/operations/cuOperations.cuh"
#include "../optimizers/Optimizers.cuh"
#include "../containers/NetParam.cuh"

namespace seann {
    
    class OperandBase {
    public:
        Parameter* X{};  //input, a shadow of the output of prev operand
        Parameter* Y{};  //output
        
        OperandBase* prev{};
        OperandBase* next{};
        
        //calculate : X -> Y
        virtual void forward() = 0;
        
        //calculate grads for operand parameters : weights, bias, etc
        virtual void paramGrads() = 0;
        
        //calculate grads for operand input : X
        virtual void xGrads() = 0;
        
        //do the gradient decent with optimizers
        virtual void updateParams() = 0;
        virtual void batchUpdateParams() = 0;
        
        virtual void initNetParams(OptimizerInfo* info, shape4 inShape) = 0;
        
        virtual void randFillNetParams() = 0;
        
        virtual void zeroGrads() = 0;
        
        virtual string info() = 0;
        
        //X should be bind to the Y of the previous operand
        void bindPrev(OperandBase* prevPtr) {
            this->prev = prevPtr;
            X->inherit(prevPtr->Y);
        }
        
        void bindNext(OperandBase* nextPtr) {
            this->next = nextPtr;
        }
        
        //grab the operands earlier in the network
        OperandBase* tracePrev(uint32 ago){
            return ago <= 0 ? this : prev->tracePrev(ago - 1);
        }
        
        OperandBase* traceNext(uint32 dist){
            return dist <= 0 ? this : next->traceNext(dist-1);
        }
    };
    
} // seann

#endif //CRUSADER_OPERANDBASE_CUH
