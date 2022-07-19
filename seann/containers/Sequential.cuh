//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_SEQUENTIAL_CUH
#define CRUSADER_SEQUENTIAL_CUH

#include "NetParam.cuh"
#include "../operands/OperandBase.cuh"
#include "../../seio/data/Dataset.cuh"

namespace seann{
    class Sequential {
    public:
        Parameter* netX;
        Parameter* netY{};
        LossFunc loss{};
        LossFuncCalc lossFW{};
        
        //a list of pointers to all the initialized operands
        OperandBase** operands{};
        uint32 OPERAND_COUNT;
        
        Sequential(shape4 inputShape, std::initializer_list<OperandBase*> list){
            OPERAND_COUNT = list.size();
            netX = Parameter::create(inputShape);
            cudaMallocHost(&operands, OPERAND_COUNT * sizeof(OperandBase*));
            for(auto i = 0; i < OPERAND_COUNT; i++) {
                operands[i] = list.begin()[i];
            }
        }
        
        //constructing from file
        Sequential(shape4 inputShape, uint32 operandCount, OperandBase** operands){
            OPERAND_COUNT = operandCount;
            netX = Parameter::create(inputShape);
            this->operands = operands;
        }
        
        void waive() const;
        
        void construct(OptimizerInfo* info);
        
        void setLoss(LossFunc loss, LossFuncCalc lossFW);
        
        void randInit() const;
        
        Tensor* forward() const;
        
        Tensor* forward(Tensor* X) const;
    
        Tensor* inferenceForward() const;
    
        Tensor* inferenceForward(Tensor *X) const;
    
        Tensor* backward(Tensor* labelY) const;
        
        void learn() const;
        
        void learnBatch() const;
        
        void train(Dataset* data, bool WITH_TEST, uint32 TEST_FREQUENCY);
    };
}

#endif //CRUSADER_SEQUENTIAL_CUH
