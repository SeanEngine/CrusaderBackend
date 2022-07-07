//
// Created by Dylan on 7/4/2022.
//

#ifndef CRUSADER_DENSEBLOCK_CUH
#define CRUSADER_DENSEBLOCK_CUH

#include "../OperandBase.cuh"
#include "../seblas/BatchNorm.cuh"
#include "../seblas/ReLU.cuh"
#include "../cudnn/cuConv2D.cuh"
#include "../controlling/ChannelConcatenater.cuh"

namespace seann {
    
    class DenseBlock : public OperandBase {
    public:
        OperandBase** operands{};
        uint32 operandCount;
        
        //dense block construction configurations
        uint32 k;
        uint32 l;
        cudnnHandle_t cudnn;
        
        DenseBlock( cudnnHandle_t cudnn, uint32 k, uint32 l){
            this->k = k;
            this->l = l;
            this->cudnn = cudnn;
            
            //BN-ReLU-Conv-Concat
            this->operandCount = 7 * l;
            cudaMallocHost(&operands, sizeof(OperandBase*) * operandCount);
        }
        
        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            X = Parameter::declare(inShape);
            uint32 k0 = inShape.c;
            uint32 n = inShape.n;

            //initialize block operands
            for(uint32 comp = 0; comp < l; comp++){
                
                operands[comp*7] = new BatchNorm();
                operands[comp*7 + 1] = new ReLU();
                //1x1 bottleneck
                operands[comp*7 + 2] = new cuConv2D(
                        cudnn,
                        shape4(4 * k, k0 + k * comp, 1, 1),
                        1, 1, 0, 0, false
                        );
                operands[comp*7 + 3] = new BatchNorm();
                operands[comp*7 + 4] = new ReLU();
                //3x3 convolution
                operands[comp*7 + 5] = new cuConv2D(
                        cudnn,
                        shape4(k, 4 * k, 3, 3),
                        1, 1, 1, 1, false
                        );
                operands[comp*7 + 6] = new ChannelConcatenater(2, k0 + k * (comp+1));
            }
            
            //initialize block parameters
            operands[0]->initNetParams(info, shape4(n, k0, inShape.h, inShape.w));
            for(uint32 i = 1; i < l * 7; i++){
                operands[i]->initNetParams(info, operands[i-1]->Y->A->dims);
            }
            
            //waive
            operands[0]->bindInput(X);
            for(uint32 i = 1; i < l * 7; i++){
                operands[i]->bindPrev(operands[i-1]);
                operands[i]->bindInput(operands[i-1]->Y);
            }
            
            //bind the concatenaters
            ((ChannelConcatenater*)operands[6])->Xs[1] = X;
            for(uint32 comp = 1; comp < l; comp++){
                ((ChannelConcatenater*)operands[comp*7 + 6])->Xs[1] = operands[comp*7 - 1]->Y;
            }
        }
        
        void forward() override;
        
        void xGrads() override;
        
        void paramGrads() override;
        
        void updateParams() override;
        
        void batchUpdateParams() override;
        
        void zeroGrads() override;
        
        string info() override;
        
        uint32 OPERAND_ID() override {
            return 0xb001;
        }
    };
    
} // seann

#endif //CRUSADER_DENSEBLOCK_CUH
