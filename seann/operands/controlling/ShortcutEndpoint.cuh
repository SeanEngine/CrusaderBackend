//
// Created by Dylan on 6/29/2022.
//

#ifndef CRUSADER_SHORTCUTENDPOINT_CUH
#define CRUSADER_SHORTCUTENDPOINT_CUH

#include "../OperandBase.cuh"

namespace seann {
    
    class ShortcutEndpoint : public OperandBase {
    public:
        bool isContainer;
        ShortcutEndpoint* other{};
        OperandBase** branchOperands{};
        uint32 operandCount = 0;
        uint32 uuid;
        ShortcutEndpoint(bool isContainer, uint32 uuid,
                            std::initializer_list<OperandBase*> operands){
            this->isContainer = isContainer;
            this->uuid = uuid;
            
            if(isContainer && operands.size() > 0){
                this->operandCount = operands.size();
                cudaMallocHost(&branchOperands, operandCount * sizeof(OperandBase *));
                for (auto i = 0; i < operandCount; i++) {
                    branchOperands[i] = operands.begin()[i];
                }
            }
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override;
        
        void postWaiveInit(OptimizerInfo *inf) override;
        
        void forward() override;
        
        void xGrads() override;
        
        void paramGrads() override;
        
        void updateParams() override;
        
        void batchUpdateParams() override;
        
        void randFillNetParams() override;
        
        void zeroGrads() override;
        
        string info() override{
            return "ShortcutEndpoint { " + std::to_string(operandCount) + " }";
        }
        
        uint32 OPERAND_ID() override{
            if(isContainer)
                return 0xf002;
            else
                return 0xf001;
        }
    };
} // seann

#endif //CRUSADER_SHORTCUTENDPOINT_CUH
