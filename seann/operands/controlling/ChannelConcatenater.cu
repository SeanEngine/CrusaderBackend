//
// Created by Dylan on 7/4/2022.
//

#include "ChannelConcatenater.cuh"
#include "../../../seblas/assist/Inspections.cuh"

namespace seann {
    void ChannelConcatenater::forward() {
        concat(Xs, paramCount, Y);
    }
    
    void ChannelConcatenater::xGrads() {
        *Y->dA + Y->dAReserve;
        concatGrads(Y, Xs, paramCount);
        cudaMemcpy(Xs[0]->dA->elements, Xs[0]->dAReserve->elements,
                   Xs[0]->dA->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
        Y->dAReserve->constFill(0);
        Xs[0]->dAReserve->constFill(0);
    }
    
    uint32 ChannelConcatenater::encodeInfo(fstream *fout, uint64 offset) {
        fout->seekp((long long) offset);
        fout->write((char*)&paramCount, sizeof(uint32));
        fout->write((char*)&outputChannels, sizeof(uint32));
        if (paramCount > 1) {
            fout->write((char *) &locations, sizeof(uint32) * (paramCount - 1));
            return sizeof(uint32) * (2 + paramCount - 1);
        }
        return sizeof(uint32) * 2;
    }
    
    uint32 ChannelConcatenater::encodeNetParams(fstream *fout, uint64 offset) {
        return 0;
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
} // seann