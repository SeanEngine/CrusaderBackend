//
// Created by Dylan on 7/17/2022.
//

#include "codec.cuh"

namespace seio{
    void saveSequence(const char* fname, Sequential* seq, uint32 epochID){
        
        auto foutObj = fstream(fname, ios::binary | ios::app);
        auto* fout = &foutObj;
        logInfo(LOG_SEG_SEIO, string(fname));
        fout->seekp(0, ios::beg);
        
        //generate the file header
        FILE_HEAD header = *new FILE_HEAD(seq->OPERAND_COUNT, epochID);
        
        //create file pointer tracker
        uint64 offset = 0;
        offset += header.encode(fout);
        
        //encode operand ids
        fout->seekp((long long) offset);
        for (uint32 i = 0; i < seq->OPERAND_COUNT; i++){
            uint32 temp = seq->operands[i]->OPERAND_ID();
            fout->write((char*)&temp, sizeof(uint32));
            offset += sizeof(uint32);
        }
        
        //encode operand info
        for (uint32 i = 0; i < seq->OPERAND_COUNT; i++){
            offset += seq->operands[i]->encodeInfo(fout, offset);
        }
        
        //encode operand parameters
        for (uint32 i = 0; i < seq->OPERAND_COUNT; i++){
            offset += seq->operands[i]->encodeNetParams(fout, offset);
        }
    }
    
    Sequential* loadSequence(const char* fname, shape4 inShape, OptimizerInfo* info){
        auto finObj = fstream(fname, ios::binary | ios::in);
        auto* fin = &finObj;
        
        //read the file header
        FILE_HEAD header = *new FILE_HEAD(0, 0);
        header.decode(fin);
        logDebug(LOG_SEG_SEIO, "Loading model from " + string(fname));
        logDebug(LOG_SEG_SEIO, "Operand count: " + to_string(header.OPERAND_COUNT)
        + " Epoch ID: " + to_string(header.EPOCH_ID));
        
        //create the offset indicator (skipped header)
        uint64 offset = 12;
        
        //read the operand ids and initialize operand objects
        OperandBase** operands;
        cudaMallocHost(&operands, sizeof(OperandBase*) * header.OPERAND_COUNT);
        assertCuda(__FILE__, __LINE__);
        uint64 operandInfoOffset = offset + sizeof(uint32) * header.OPERAND_COUNT;
        
        for (uint32 i = 0; i < header.OPERAND_COUNT; i++){
            uint32 temp = 0;
            fin->seekg((long long) offset);
            fin->read((char*)&temp, sizeof(uint32));
            offset += sizeof(uint32);
            
            //locate operand decoder of the type
            OperandBase* operand = getInfoDecoder(temp)(fin, operandInfoOffset);
            operands[i] = operand;
            logDebug(LOG_SEG_SEIO, "Loaded operand with type : " + to_string(temp));
        }
    
        logDebug(LOG_SEG_SEIO,"Operand info decoding done...");
        
        //read the operand parameters
        offset = operandInfoOffset;
        for (uint32 i = 0; i < header.OPERAND_COUNT; i++){
            getParamDecoder(operands[i]->OPERAND_ID())(fin, offset, operands[i],
                    info, i > 0 ? operands[i-1]->Y->A->dims : inShape);
            logInfo(LOG_SEG_SEIO, operands[i]->info());
            operands[i]->operandID = i;
        }
        
        auto* seq = new Sequential(inShape,  header.OPERAND_COUNT, operands);
        //bind inputs and outputs
        operands[0]->X->inherit(seq->netX);
        seq->netY = Parameter::declare(operands[seq->OPERAND_COUNT - 1]->Y->A->dims)
                ->inherit(operands[seq->OPERAND_COUNT - 1]->Y);
    
        //bind the layers together
        seq->waive();
    
        for(int i = 0; i < seq->OPERAND_COUNT; i++){
            operands[i]->postWaiveInit(info);
        }
        return seq;
    }
}