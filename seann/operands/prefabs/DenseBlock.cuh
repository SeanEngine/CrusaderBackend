//
// Created by Dylan on 7/4/2022.
//

#ifndef CRUSADER_DENSEBLOCK_CUH
#define CRUSADER_DENSEBLOCK_CUH

#include "../OperandBase.cuh"
#include "../seblas/BatchNorm.cuh"
#include "../seblas/activate/ReLU.cuh"
#include "../cudnn/cuConv2D.cuh"
#include "../controlling/ChannelConcatenater.cuh"

#define OPR_PREFAB_DENSEBLOCK 0xe001

namespace seann {
    
    class DenseBlock : public OperandBase {
    public:
        OperandBase** operands{};
        uint32 operandCount;
        
        //dense block construction configurations
        uint32 k;
        uint32 l;
        
        DenseBlock(uint32 k, uint32 l){
            this->k = k;
            this->l = l;
            
            //BN-ReLU-Conv-Concat
            this->operandCount = 7 * l + 1;
            cudaMallocHost(&operands, sizeof(OperandBase*) * operandCount);
        }
        
        void initNetFrame(OptimizerInfo *info, shape4 inShape){
            uint32 k0 = inShape.c;
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape.n, k0 + k * l, inShape.h, inShape.w);
    
            //initialize block operands
            for(uint32 comp = 0; comp < l; comp++){
                if(comp==0) {
                    operands[comp * 7] = new ChannelConcatenater(1, k0);
                }else{
                    operands[comp * 7] = new ChannelConcatenater(2, k0 + k * (comp), {7});
                }
                operands[comp*7 + 1] = new BatchNorm();
                operands[comp*7 + 2] = new ReLU();
                //1x1 bottleneck
                operands[comp*7 + 3] = new cuConv2D(
                        shape4(4 * k, k0 + k * comp, 1, 1),
                        1, 1, 0, 0, false
                );
                operands[comp*7 + 4] = new BatchNorm();
                operands[comp*7 + 5] = new ReLU();
                //3x3 convolution
                operands[comp*7 + 6] = new cuConv2D(
                        shape4(k, 4 * k, 3, 3),
                        1, 1, 1, 1, false
                );
            }
    
            operands[operandCount-1] = new ChannelConcatenater(2, k0 + k * l, {7});
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override{

            uint32 k0 = inShape.c;
            uint32 n = inShape.n;
    
            initNetFrame(info, inShape);
            
            //initialize block parameters
            operands[0]->initNetParams(info, shape4(n, k0, inShape.h, inShape.w));
            for(uint32 i = 1; i < operandCount; i++){
                operands[i]->initNetParams(info, operands[i-1]->Y->A->dims);
            }
        }
        
        void postWaiveInit(OptimizerInfo *inf) override{
            //waive
            operands[0]->bindInput(X);
            for(uint32 i = 1; i < operandCount; i++){
                operands[i]->bindPrev(operands[i-1]);
                operands[i]->bindInput(operands[i-1]->Y);
                operands[i-1]->bindNext(operands[i]);
            }
            
            for(uint32 i = 0; i < operandCount; i++){
                operands[i]->postWaiveInit(inf);
            }
        }
        
        void forward() override;
        
        void xGrads() override;
        
        void paramGrads() override;
        
        void updateParams() override;
        
        void batchUpdateParams() override;
        
        void randFillNetParams() override;
        
        void zeroGrads() override;
        
        string info() override;
        
        uint32 OPERAND_ID() override {
            return OPR_PREFAB_DENSEBLOCK;
        }
        
        float getOptimLR() override{
            //since conv layer has build in optimizers
            return operands[3]->getOptimLR();
        }
        
        float getL2Const() override{
            return operands[3]->getL2Const();
        }
        
        void updateOptimLR(float val) override{
            for(uint32 i = 0; i < operandCount; i++){
                operands[i]->updateOptimLR(val);
            }
        }
        
        void updateL2Const(float val) override{
            for(uint32 i = 0; i < operandCount; i++){
                operands[i]->updateL2Const(val);
            }
        }
        
        uint32 encodeInfo(fstream *fout, uint64 offset) override;
        
        uint32 encodeNetParams(fstream *fout, uint64 offset) override;
        
        uint32 getInfoEncodingSize() override{
            uint32 size = 0;
            for(uint32 i = 0; i < operandCount; i++){
                size += operands[i]->getInfoEncodingSize();
            }
            return sizeof(uint32) * 2 + size;
        }
        
        uint32 getNetParamsEncodingSize() override{
            uint32 size = 0;
            for(uint32 i = 0; i < operandCount; i++){
                size += operands[i]->getNetParamsEncodingSize();
            }
            return size;
        }
    };
    
} // seann

#endif //CRUSADER_DENSEBLOCK_CUH
