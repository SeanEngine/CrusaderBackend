//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_DATALOADER_CUH
#define CRUSADER_DATALOADER_CUH

#include "Data.cuh"
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../fileParsers/pugixml.cuh"

#define DTYPE_UCHAR 0x00
#define DTYPE_FLOAT_32 0x01

using namespace cv;
namespace seio {
    enum BBLabelType{
        BB_YOLO_1
    };
    
    void readBytes(unsigned char *buffer, unsigned long size, const char* binPath);
    
    unsigned long getFileSize(const char* binPath);
    
    void fetchCrdat(Tensor* x, Tensor* label, const char* rootPath, const char* datasetName,
                    uint32 offset, unsigned char** buffer);
    
} // seann

#endif //CRUSADER_DATALOADER_CUH
