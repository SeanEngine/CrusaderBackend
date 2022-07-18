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
        runningOffset += sizeof(uint32) * 5;
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
            fin->read((char*)&locations, sizeof(uint32) * (paramCount - 1));
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
        shortcut->initNetParams(info, inShape);
    }
}
