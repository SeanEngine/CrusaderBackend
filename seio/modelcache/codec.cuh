//
// Created by Dylan on 7/17/2022.
//

#ifndef CRUSADER_CODEC_CUH
#define CRUSADER_CODEC_CUH

#include "../../seann/seann.cuh"

namespace seio{
    
    /**
     * FILE ARCHITECTURE
     *
     * header:
     * BYTE: 0x00 - 0x04: magic number
     * BYTE: 0x05 - 0x09: version number
     * BYTE: 0x0A - 0x0D: number of layers
     * BYTE: 0x0E - 0x11: epochID
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
         uint32 FILE_SIZE = 0;
         uint32 OPERAND_COUNT = 0;
         uint32 EPOCH_ID = 0;
    };
    
    
}


#endif //CRUSADER_CODEC_CUH
