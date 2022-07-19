//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_OPERANDBASE_CUH
#define CRUSADER_OPERANDBASE_CUH

#include "../../seblas/operations/cuOperations.cuh"
#include "../optimizers/Optimizers.cuh"
#include "../containers/NetParam.cuh"

using namespace std;
extern cudnnHandle_t cudnn;

namespace seann {
    
    class OperandBase {
    public:
        Parameter* X;  //input, a shadow of the output of prev operand
        Parameter* Y;  //output
        
        OperandBase* prev = nullptr;
        OperandBase* next = nullptr;
        
        uint32 operandID = 0;
        
        OperandBase(){
            if (cudnn == nullptr) {
                cudnnCreate(&cudnn);
            }
        }
        
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
        
        virtual void inferenceForward(){
            forward();
        };
        
        virtual void zeroGrads() = 0;
        
        virtual string info() = 0;
        
        //X should be bind to the Y of the previous operand
        void bindPrev(OperandBase* prevPtr) {
            this->prev = prevPtr;
        }
        
        void bindNext(OperandBase* nextPtr) {
            this->next = nextPtr;
        }
        
        void bindInput(Parameter* prevY) const{
            this->X->inherit(prevY);
        }
        
        //grab the operands earlier in the network
        OperandBase* tracePrev(uint32 ago){
            return ago <= 0 ? this : prev->tracePrev(ago - 1);
        }
        
        OperandBase* traceNext(uint32 dist){
            return dist <= 0 ? this : next->traceNext(dist-1);
        }
    
        OperandBase* tracePrev() const{
            return prev;
        }
        
        OperandBase* traceNext() const{
            return next;
        }
        
        virtual uint32 OPERAND_ID(){
            return 0x0000;
        }
        
        virtual void postWaiveInit(OptimizerInfo* inf){}
        
        virtual float getOptimLR(){return 0;}
        
        virtual void updateOptimLR(float val){}
        
        virtual float getL2Const(){return 0;}
        
        virtual void updateL2Const(float val){}
        
        virtual uint32 encodeInfo(fstream* fout, uint64 offset) = 0;
        
        virtual uint32 encodeNetParams(fstream* fout, uint64 offset) = 0;
        
        virtual uint32 getInfoEncodingSize() = 0;
        
        virtual uint32 getNetParamsEncodingSize() = 0;
    };
    
} // seann

#endif //CRUSADER_OPERANDBASE_CUH
