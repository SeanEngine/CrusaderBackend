//
// Created by Dylan on 7/4/2022.
//

#ifndef CRUSADER_CHANNELCONCATENATER_CUH
#define CRUSADER_CHANNELCONCATENATER_CUH

#include "../OperandBase.cuh"

namespace seann {
    class ChannelConcatenater : public OperandBase {
    public:
        uint32 paramCount;
        uint32 outputChannels;
        Parameter** Xs{};
        
        explicit ChannelConcatenater(uint32 paramCount, uint32 outputChannels) : OperandBase() {
            this->paramCount = paramCount;
            cudaMallocHost(&Xs, sizeof(Parameter*) * paramCount);
            this->outputChannels = outputChannels;
        }
        
        void bindParams(std::initializer_list<Parameter*> ps) const {
            assert(ps.size() == paramCount - 1);
            for (auto i = 1; i < paramCount; i++) {
                this->Xs[i] = ps.begin()[i-1];
            }
        }
        
        void bindParams(Parameter** Xs0, uint32 paramCount0) const {
            assert(paramCount0 == paramCount - 1);
            for (auto i = 1; i < paramCount; i++) {
                this->Xs[i] = Xs0[i-1];
            }
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape.n, outputChannels, inShape.w, inShape.h);
            Xs[0] = X;
        }
        
        void forward() override;
        
        void xGrads() override;
        
        void paramGrads() override{}
        
        void updateParams() override{}
        
        void batchUpdateParams() override{}
        
        void randFillNetParams() override{}
        
        void zeroGrads() override{}
        
        uint32 OPERAND_ID() override {
            return 0xf003;
        }
        
        string info() override {
            return "ChannelConcatenater   { " + std::to_string(paramCount) + " }";
        }
    };
} // seann

#endif //CRUSADER_CHANNELCONCATENATER_CUH
