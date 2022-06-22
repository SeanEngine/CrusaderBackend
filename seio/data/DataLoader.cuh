//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_DATALOADER_CUH
#define CRUSADER_DATALOADER_CUH

#include "Dataset.cuh"
#include <cstdlib>
#include <fstream>

namespace seio {
    typedef unsigned char BYTE;
    
    void readBytes(BYTE *buffer, unsigned long size, const char* binPath);
    
    unsigned long getFileSize(const char* binPath);
    
    Dataset* fetchIDX(Dataset* dataset, const char* binPath, uint32 step, bool isLabel);
    
    //<1 x label><3072 x pixel>
    //...
    //<1 x label><3072 x pixel>
    Dataset* fetchCIFAR(Dataset* dataset, const char* binPath, uint32 fileID);
    
} // seann

#endif //CRUSADER_DATALOADER_CUH
