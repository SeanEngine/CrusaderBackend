//
// Created by Dylan on 7/18/2022.
//

#ifndef CRUSADER_CODECREGISTRY_CUH
#define CRUSADER_CODECREGISTRY_CUH

#include "containers/Sequential.cuh"
#include "operands/seblas/Conv2D.cuh"
#include "operands/seblas/Linear.cuh"
#include "operands/seblas/activate/ReLU.cuh"
#include "operands/seblas/activate/LReLU.cuh"
#include "operands/seblas/Softmax.cuh"
#include "operands/seblas/pooling/MaxPool2D.cuh"
#include "operands/seblas/BatchNorm.cuh"
#include "operands/seblas/pooling/AvgPool2D.cuh"
#include "optimizers/Optimizers.cuh"
#include "operands/seblas/Dropout.cuh"
#include "operands/controlling/ShortcutEndpoint.cuh"
#include "operands/seblas/pooling/GlobalAvgPool.cuh"

#include "operands/prefabs/DenseBlock.cuh"

#include "operands/cudnn/cuConv2D.cuh"

namespace seann{
    OperandBase::InfoDecoder getInfoDecoder(uint32 OPERAND_ID){
        switch(OPERAND_ID){
            case OPR_SEBLAS_SOFTMAX : return DEC_OPR_SOFTMAX_INFO;
            case OPR_SEBLAS_CONV2D : return DEC_OPR_CONV2D_INFO;
            case OPR_SEBLAS_LINEAR : return DEC_OPR_LINEAR_INFO;
            case OPR_SEBLAS_RELU : return DEC_OPR_RELU_INFO;
            case OPR_SEBLAS_LRELU : return DEC_OPR_LRELU_INFO;
            case OPR_SEBLAS_MAXPOOL2D : return DEC_OPR_MAXPOOL2D_INFO;
            case OPR_SEBLAS_AVGPOOL2D : return DEC_OPR_AVGPOOL2D_INFO;
            case OPR_SEBLAS_BATCHNORM : return DEC_OPR_BATCHNORM_INFO;
            case OPR_SEBLAS_DROPOUT : return DEC_OPR_DROPOUT_INFO;
            case OPR_CTRL_SHORTCUTENDPOINT_BEG : return DEC_OPR_SHORTCUTENDPOINT_BEG_INFO;
            case OPR_CTRL_SHORTCUTENDPOINT_END : return DEC_OPR_SHORTCUTENDPOINT_END_INFO;
            case OPR_SEBLAS_GLOBALAVGPOOL : return DEC_OPR_GLOBALAVGPOOL_INFO;
            case OPR_CTRL_CHANNELCONCATENATER : return DEC_OPR_CHANNELCONCATENATER_INFO;
            case OPR_CUDNN_CONV2D : return DEC_OPR_CONV2D_INFO;
            default: return nullptr;
        }
    }
}

#endif //CRUSADER_CODECREGISTRY_CUH
