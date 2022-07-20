//
// Created by Dylan on 7/18/2022.
//

#include "OperandCodecs.cuh"

namespace seio{
    OperandBase* DEC_OPR_SOFTMAX_INFO(fstream* fin, uint64& offset) {
        return new Softmax();
    }
    
    void DEC_OPR_SOFTMAX_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_LINEAR_INFO(fstream* fin, uint64& offset){
        uint32 OUTPUT_SIZE;
        fin->seekg((long long)offset);
        fin->read((char*)&OUTPUT_SIZE, sizeof(uint32));
        auto* opr = new Linear(OUTPUT_SIZE);
        offset += sizeof(uint32);
        return opr;
    }
    
    void DEC_OPR_LINEAR_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape){
        auto* linear = (Linear*)opr;
        linear->initNetParams(info, inShape);
        NetParam::decodeNetParamData(fin, offset, linear->weights);
        NetParam::decodeNetParamData(fin, offset, linear->biases);
    }
    
    OperandBase* DEC_OPR_DROPOUT_INFO(fstream* fin, uint64& offset) {
        float p;
        fin->seekg((long long)offset);
        fin->read((char*)&p, sizeof(float));
        offset += sizeof(float);
        return new Dropout(p);
    }
    
    void DEC_OPR_DROPOUT_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
    OperandBase *DEC_OPR_CONV2D_INFO(fstream *fin, uint64& offset) {
        uint32 runningOffset = offset;
        uint32 strideH, strideW, padH, padW, n;
        uint32 fn, fc, fh, fw;
        fin->seekg(runningOffset);
        fin->read((char*)&fn, sizeof(uint32));
        fin->read((char*)&fc, sizeof(uint32));
        fin->read((char*)&fh, sizeof(uint32));
        fin->read((char*)&fw, sizeof(uint32));
        
        fin->read((char*)&strideH, sizeof(uint32));
        fin->read((char*)&strideW, sizeof(uint32));
        fin->read((char*)&padH, sizeof(uint32));
        fin->read((char*)&padW, sizeof(uint32));
        fin->read((char*)&n, sizeof(uint32));
        runningOffset += sizeof(uint32) * 9;
        offset = runningOffset;
        
        return new Conv2D(shape4(fn, fc, fh, fw), strideH, strideW, padH, padW,
                          n > 0);
    }
    
    void DEC_OPR_CONV2D_PARAM(fstream *fin, uint64& offset, OperandBase *opr, OptimizerInfo* info, shape4 inShape) {
        uint64 runningOffset = offset;
        auto *conv = (Conv2D*)opr;
        conv->initNetParams(info, inShape);
        NetParam::decodeNetParamData(fin, runningOffset, conv->filter);
        if(conv->WITH_BIAS){
            NetParam::decodeNetParamData(fin, runningOffset, conv->bias);
        }
        offset = runningOffset;
    }
    
    OperandBase* DEC_OPR_BATCHNORM_INFO(fstream *fout, uint64& offset) {
        return new BatchNorm();
    }
    
    void DEC_OPR_BATCHNORM_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        auto* bn = (BatchNorm*)opr;
        bn->initNetParams(info, inShape);
        NetParam::decodeNetParamData(fin, offset, bn->beta);
        NetParam::decodeNetParamData(fin, offset, bn->gamma);
    }
    
    OperandBase* DEC_OPR_MAXPOOL2D_INFO(fstream* fin, uint64& offset) {
        uint32 strideH, strideW, rangeH, rangeW;
        fin->seekg((long long) offset);
        fin->read((char*)&strideH, sizeof(uint32));
        fin->read((char*)&strideW, sizeof(uint32));
        fin->read((char*)&rangeH, sizeof(uint32));
        fin->read((char*)&rangeW, sizeof(uint32));
        offset += sizeof(uint32) * 4;
        return new MaxPool2D(strideH, strideW, rangeH, rangeW);
    }
    
    void DEC_OPR_MAXPOOL2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
    
    OperandBase* DEC_OPR_GLOBALAVGPOOL_INFO(fstream* fin, uint64& offset) {
        return new GlobalAvgPool();
    }
    
    void DEC_OPR_GLOBALAVGPOOL_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_AVGPOOL2D_INFO(fstream* fin, uint64& offset) {
        uint32 strideH, strideW, rangeH, rangeW;
        fin->seekg((long long)offset);
        fin->read((char*)&strideH, sizeof(uint32));
        fin->read((char*)&strideW, sizeof(uint32));
        fin->read((char*)&rangeH, sizeof(uint32));
        fin->read((char*)&rangeW, sizeof(uint32));
        offset += sizeof(uint32) * 4;
        return new AvgPool2D(strideH, strideW, rangeH, rangeW);
    }
    
    void DEC_OPR_AVGPOOL2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_LRELU_INFO(fstream* fin, uint64& offset) {
        float alpha;
        fin->seekg((long long) offset);
        fin->read((char*)&alpha, sizeof(float));
        offset += sizeof(float);
        return new LReLU(alpha);
    }
    
    void DEC_OPR_LRELU_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_RELU_INFO(fstream *fout, uint64& offset) {
        return new ReLU();
    }
    
    void DEC_OPR_RELU_PARAM(fstream *fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        opr->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_CUDNN_CONV2D_INFO(fstream* fin, uint64& offset){
        uint32 runningOffset = offset;
        uint32 strideH, strideW, padH, padW, n;
        uint32 fn, fc, fh, fw;
        fin->seekg(runningOffset);
        fin->read((char*)&fn, sizeof(uint32));
        fin->read((char*)&fc, sizeof(uint32));
        fin->read((char*)&fh, sizeof(uint32));
        fin->read((char*)&fw, sizeof(uint32));
        
        fin->read((char*)&strideH, sizeof(uint32));
        fin->read((char*)&strideW, sizeof(uint32));
        fin->read((char*)&padH, sizeof(uint32));
        fin->read((char*)&padW, sizeof(uint32));
        fin->read((char*)&n, sizeof(uint32));
        runningOffset += sizeof(uint32) * 9;
        offset = runningOffset;
        
        return new cuConv2D(shape4(fn, fc, fh, fw), strideH, strideW, padH, padW,
                            n > 0);
    }
    
    void DEC_OPR_CUDNN_CONV2D_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape){
        auto* conv = (cuConv2D*)opr;
        conv->initNetParams(info, inShape);
        NetParam::decodeNetParamData(fin, offset, conv->filter);
        if(conv->WITH_BIAS){
            NetParam::decodeNetParamData(fin, offset, conv->bias);
        }
    }
    
    OperandBase* DEC_OPR_CHANNELCONCATENATER_INFO(fstream* fin, uint64& offset) {
        uint32 paramCount;
        uint32 outputChannels;
        fin->seekg((long long) offset);
        fin->read((char*)&paramCount, sizeof(uint32));
        fin->read((char*)&outputChannels, sizeof(uint32));
        offset += sizeof(uint32) * 2;
        if (paramCount > 1) {
            uint32* locations;
            cudaMallocHost(&locations, sizeof(uint32) * (paramCount - 1));
            assertCuda(__FILE__, __LINE__);
            
            fin->read((char*)locations, sizeof(uint32) * (paramCount - 1));
            offset += sizeof(uint32) * (paramCount - 1);
            return new ChannelConcatenater(paramCount, outputChannels, locations);
        } else {
            return new ChannelConcatenater(paramCount, outputChannels);
        }
    }
    
    void DEC_OPR_CHANNELCONCATENATER_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape) {
        auto* concat = (ChannelConcatenater*) opr;
        concat->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_SHORTCUTENDPOINT_BEG_INFO(fstream* fin, uint64& offset){
        uint32 uuid;
        fin->seekg((long long) offset);
        fin->read((char *) &uuid, sizeof(uint32));
        offset += sizeof(uint32);
        uint32 operandCount;
        fin->read((char *) &operandCount, sizeof(uint32));
        offset += sizeof(uint32);
        return new ShortcutEndpoint(false, uuid, {});
    }
    
    void DEC_OPR_SHORTCUTENDPOINT_BEG_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape){
        auto* shortcut = (ShortcutEndpoint*)opr;
        shortcut->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_SHORTCUTENDPOINT_END_INFO(fstream* fin, uint64& offset){
        uint32 uuid;
        fin->seekg((long long)offset);
        fin->read((char *) &uuid, sizeof(uint32));
        offset += sizeof(uint32);
        uint32 operandCount;
        fin->read((char *) &operandCount, sizeof(uint32));
        offset += sizeof(uint32);
        
        uint64 subInfoOffset = offset + sizeof(uint32) * operandCount;
        
        if(operandCount > 0) {
            OperandBase **branchOperands;
            cudaMallocHost(&branchOperands, sizeof(OperandBase *) * operandCount);
            for (uint32 i = 0; i < operandCount; i++) {
                fin->seekg((long long) offset);
                uint32 temp;
                fin->read((char *) &temp, sizeof(uint32));
                offset += sizeof(uint32);
                auto decoder = getInfoDecoder(temp);
                branchOperands[i] = decoder(fin, subInfoOffset);
            }
            return new ShortcutEndpoint(true, uuid, branchOperands, operandCount);
        }
        return new ShortcutEndpoint(true, uuid, {});
    }
    
    void DEC_OPR_SHORTCUTENDPOINT_END_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape){
        auto* shortcut = (ShortcutEndpoint*)opr;
        for(uint32 i = 0; i < shortcut->operandCount; i++){
            shape4 optShape = i > 0 ? inShape : shortcut->branchOperands[i-1]->Y->A->dims;
            uint32 execId = shortcut->branchOperands[i]->OPERAND_ID();
            auto decode = getParamDecoder(execId);
            decode(fin, offset, shortcut->branchOperands[i], info, optShape);
        }
        shortcut->initNetParams(info, inShape);
    }
    
    OperandBase* DEC_OPR_DENSEBLOCK_INFO(fstream* fin, uint64& offset){
        uint32 l, k;
        fin->seekg((long long) offset);
        fin->read((char *) &l, sizeof(uint32));
        fin->read((char *) &k, sizeof(uint32));
        offset += sizeof(uint32) * 2;
        return new DenseBlock(k,l);
    }
    
    void DEC_OPR_DENSEBLOCK_PARAM(fstream* fin, uint64& offset, OperandBase* opr, OptimizerInfo* info, shape4 inShape){
        auto* dense = (DenseBlock*)opr;
        dense->initNetFrame(info, inShape);
        for (uint32 i = 0; i < dense->operandCount; i++){
            auto temp = dense->operands[i]->OPERAND_ID();
            auto decode = getParamDecoder(temp);
            decode(fin, offset, dense->operands[i], info,  i <= 0 ? inShape
             : dense->operands[i-1]->Y->A->dims);
        }
    }
    
    InfoDecoder getInfoDecoder(uint32 OPERAND_ID){
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
            case OPR_CUDNN_CONV2D : return DEC_OPR_CUDNN_CONV2D_INFO;
            case OPR_PREFAB_DENSEBLOCK : return DEC_OPR_DENSEBLOCK_INFO;
            default: return nullptr;
        }
    }
    
    ParamDecoder getParamDecoder(uint32 OPERAND_ID){
        switch(OPERAND_ID){
            case OPR_SEBLAS_SOFTMAX : return DEC_OPR_SOFTMAX_PARAM;
            case OPR_SEBLAS_CONV2D : return DEC_OPR_CONV2D_PARAM;
            case OPR_SEBLAS_LINEAR : return DEC_OPR_LINEAR_PARAM;
            case OPR_SEBLAS_RELU : return DEC_OPR_RELU_PARAM;
            case OPR_SEBLAS_LRELU : return DEC_OPR_LRELU_PARAM;
            case OPR_SEBLAS_MAXPOOL2D : return DEC_OPR_MAXPOOL2D_PARAM;
            case OPR_SEBLAS_AVGPOOL2D : return DEC_OPR_AVGPOOL2D_PARAM;
            case OPR_SEBLAS_BATCHNORM : return DEC_OPR_BATCHNORM_PARAM;
            case OPR_SEBLAS_DROPOUT : return DEC_OPR_DROPOUT_PARAM;
            case OPR_CTRL_SHORTCUTENDPOINT_BEG : return DEC_OPR_SHORTCUTENDPOINT_BEG_PARAM;
            case OPR_CTRL_SHORTCUTENDPOINT_END : return DEC_OPR_SHORTCUTENDPOINT_END_PARAM;
            case OPR_SEBLAS_GLOBALAVGPOOL : return DEC_OPR_GLOBALAVGPOOL_PARAM;
            case OPR_CTRL_CHANNELCONCATENATER : return DEC_OPR_CHANNELCONCATENATER_PARAM;
            case OPR_CUDNN_CONV2D : return DEC_OPR_CUDNN_CONV2D_PARAM;
            case OPR_PREFAB_DENSEBLOCK : return DEC_OPR_DENSEBLOCK_PARAM;
            default: return nullptr;
        }
    }
}
