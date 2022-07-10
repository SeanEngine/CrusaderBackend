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
        uint32* locations;
        bool useLocationGrabber = false;
        Parameter** Xs{};
        
        explicit ChannelConcatenater(uint32 paramCount, uint32 outputChannels) : OperandBase() {
            this->paramCount = paramCount;
            cudaMallocHost(&Xs, sizeof(Parameter*) * paramCount);
            this->outputChannels = outputChannels;
        }
        
        ChannelConcatenater(uint32 paramCount, uint32 outputChannels, std::initializer_list<uint32> operandOutputs)
        : ChannelConcatenater(paramCount, outputChannels){
            assert(operandOutputs.size() == paramCount - 1);
            useLocationGrabber = true;
            cudaMallocHost(&locations, sizeof(uint32) * (paramCount - 1));
            for (auto i = 0; i < paramCount - 1; i++) {
                locations[i] = operandOutputs.begin()[i];
            }
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
        }
        
        void postWaiveInit(OptimizerInfo *inf) override{
            Xs[0] = X;
            if(useLocationGrabber){
                for (auto i = 1; i < paramCount; i++) {
                    Xs[i] = tracePrev(locations[i-1])->Y;
                }
            }
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
