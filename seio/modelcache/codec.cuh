//
// Created by Dylan on 7/17/2022.
//

#ifndef CRUSADER_CODEC_CUH
#define CRUSADER_CODEC_CUH

#include "../../seann/containers/Sequential.cuh"
#include "OperandCodecs.cuh"

#include <iostream>
#include <fstream>

using namespace seann;
namespace seio{
    
    /**
     * FILE ARCHITECTURE
     *
     * header:
     * BYTE: 0x00 - 0x04: magic number
     * BYTE: 0x05 - 0x09: number of layers
     * BYTE: 0x0A - 0x0D: epochID
     *
     * operand info: 0 -> OPERAND_COUNT - 1
     * <uint32 4byte operandID 0>
     * <uint32 4byte operandID 1>
     * .....
     *
     * operand parameters:
     * <operand 0 parameters...>
     * <operand 1 parameters...>
     * .....
     *
     * data:
     * float data...
     */
    
    //read directly from model file
    struct FILE_HEAD{
         const uint32 MAGIC_NUMBER = 0x7921aedf;
         uint32 OPERAND_COUNT = 0;
         uint32 EPOCH_ID = 0;
         
         FILE_HEAD(uint32 operandCount, uint32 epochID){
             OPERAND_COUNT = operandCount;
             EPOCH_ID = epochID;
         }
         
         uint32 encode(fstream* fout){
             fout->seekp(0);
             fout->write((char*)&MAGIC_NUMBER, sizeof(uint32));
             fout->write((char*)&OPERAND_COUNT, sizeof(uint32));
             fout->write((char*)&EPOCH_ID, sizeof(uint32));
             return sizeof(uint32) * 3;
         }
         
        void decode(fstream* fin){
            uint32 temp = 0;
            fin->seekg(0);
            fin->read((char*) &temp, sizeof(uint32));
            
            if (temp != MAGIC_NUMBER){
                logError(LOG_SEG_SEIO, "File Type Error : " + to_string(fin->tellg()));
                exit(1);
            }
            fin->read((char*)&OPERAND_COUNT, sizeof(uint32));
            fin->read((char*)&EPOCH_ID, sizeof(uint32));
        }
    };
    
    void saveSequence(const char* fname, Sequential* seq, uint32 epochID);
    
    Sequential* loadSequence(const char* fname, shape4 inShape, OptimizerInfo* info);
}


#endif //CRUSADER_CODEC_CUH
