//
// Created by Dylan on 7/18/2022.
//

#ifndef CRUSADER_OPERANDCODECS_CUH
#define CRUSADER_OPERANDCODECS_CUH

#include "../../seann/containers/Sequential.cuh"
#include "../../seann/operands/seblas/Conv2D.cuh"
#include "../../seann/operands/seblas/Linear.cuh"
#include "../../seann/operands/seblas/activate/ReLU.cuh"
#include "../../seann/operands/seblas/activate/LReLU.cuh"
#include "../../seann/operands/seblas/Softmax.cuh"
#include "../../seann/operands/seblas/pooling/MaxPool2D.cuh"
#include "../../seann/operands/seblas/BatchNorm.cuh"
#include "../../seann/operands/seblas/pooling/AvgPool2D.cuh"
#include "../../seann/optimizers/Optimizers.cuh"
#include "../../seann/operands/seblas/Dropout.cuh"
#include "../../seann/operands/controlling/ShortcutEndpoint.cuh"
#include "../../seann/operands/seblas/pooling/GlobalAvgPool.cuh"

#include "../../seann/operands/prefabs/DenseBlock.cuh"
#include "../../seann/operands/cudnn/cuConv2D.cuh"

#include <iostream>
#include <fstream>

using namespace std;
using namespace seann;
namespace seio{
    typedef OperandBase* (*InfoDecoder)(fstream*, uint64&);
    typedef void (*ParamDecoder)(fstream*, uint64&, OperandBase*, OptimizerInfo*, shape4 inShape);

    OperandBase* DEC_OPR_SOFTMAX_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_SOFTMAX_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_LINEAR_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_LINEAR_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_DROPOUT_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_DROPOUT_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_CONV2D_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_CONV2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_BATCHNORM_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_BATCHNORM_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_MAXPOOL2D_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_MAXPOOL2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_GLOBALAVGPOOL_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_GLOBALAVGPOOL_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_AVGPOOL2D_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_AVGPOOL2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_LRELU_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_LRELU_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_RELU_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_RELU_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_CUDNN_CONV2D_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_CUDNN_CONV2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_CHANNELCONCATENATER_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_CHANNELCONCATENATER_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_SHORTCUTENDPOINT_BEG_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_SHORTCUTENDPOINT_BEG_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_SHORTCUTENDPOINT_END_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_SHORTCUTENDPOINT_END_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    OperandBase* DEC_OPR_DENSEBLOCK_INFO(fstream* fin, uint64& offset);
    void DEC_OPR_DENSEBLOCK_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape);
    
    InfoDecoder getInfoDecoder(uint32 OPERAND_ID);
    
    ParamDecoder getParamDecoder(uint32 OPERAND_ID);
}


#endif //CRUSADER_OPERANDCODECS_CUH
