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
    
    struct Loader{
        virtual void loadData(Tensor* X, Tensor* label, const char* rootPath, const char* dataName
            , uint32 offset) = 0;
    
        virtual thread preProcess(Tensor *X, Tensor *label) = 0;
        
        //called bufore starting preProcess thread
        virtual void prepareAsync() = 0;
    };
    
    struct CrdatLoader : public Loader {
        unsigned char *fetchBuffer = nullptr;
    
        void loadData(Tensor *X, Tensor *label, const char *rootPath, const char *datasetName,
                      uint32 offset) override;
        
        //since there are no efficiency bottleneck while processing single file
        //an async pre-process function won't be necessary
        thread preProcess(Tensor *X, Tensor *label) override{
            return {};
        }
        
        void prepareAsync() override{}
    };
    
    struct ImgLoader : public Loader {
    public:
        //buffers
        cv::Mat inputImg;
        cv::Mat inputImgCpy;
    
        void loadData(Tensor *X, Tensor *label, const char *rootPath, const char *datasetName,
                      uint32 offset) override;
        
        thread preProcess(Tensor *X, Tensor *label) override;
        
        void prepareAsync() override;
    };
    
} // seann

#endif //CRUSADER_DATALOADER_CUH
