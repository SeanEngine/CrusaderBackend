//
// Created by Dylan on 6/29/2022.
//

#ifndef CRUSADER_SHORTCUTENDPOINT_CUH
#define CRUSADER_SHORTCUTENDPOINT_CUH

#include "../OperandBase.cuh"

#define OPR_CTRL_SHORTCUTENDPOINT_BEG 0xf002
#define OPR_CTRL_SHORTCUTENDPOINT_END 0xf003

namespace seann {
    
    OperandBase* DEC_OPR_SHORTCUTENDPOINT_BEG_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_SHORTCUTENDPOINT_BEG_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_SHORTCUTENDPOINT_END_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_SHORTCUTENDPOINT_END_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
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
            
            if(isContainer){
                decodeInfo = DEC_OPR_SHORTCUTENDPOINT_END_INFO;
                decodeParams = DEC_OPR_SHORTCUTENDPOINT_END_PARAM;
            } else {
                decodeInfo = DEC_OPR_SHORTCUTENDPOINT_BEG_INFO;
                decodeParams = DEC_OPR_SHORTCUTENDPOINT_BEG_PARAM;
            }
        }
        
        ShortcutEndpoint(bool isContainer, uint32 uuid, OperandBase** operands, uint32 operandCount){
            this->isContainer = isContainer;
            this->uuid = uuid;
            
            if(isContainer && operands != nullptr){
                this->operandCount = operandCount;
                branchOperands = operands;
            }
            
            if(isContainer){
                decodeInfo = DEC_OPR_SHORTCUTENDPOINT_END_INFO;
                decodeParams = DEC_OPR_SHORTCUTENDPOINT_END_PARAM;
            } else {
                decodeInfo = DEC_OPR_SHORTCUTENDPOINT_BEG_INFO;
                decodeParams = DEC_OPR_SHORTCUTENDPOINT_BEG_PARAM;
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
            return isContainer ? OPR_CTRL_SHORTCUTENDPOINT_END : OPR_CTRL_SHORTCUTENDPOINT_BEG;
        }
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
    };
} // seann

#endif //CRUSADER_SHORTCUTENDPOINT_CUH
