//
// Created by Dylan on 6/20/2022.
//

#include "DataLoader.cuh"
#include "../../seblas/assist/Inspections.cuh"
#include <sys/stat.h>


#define toLittleEndian(i) ((i>>24)&0xFF) | ((i>>8)&0xFF00) | ((i<<8)&0xFF0000) | ((i<<24)&0xFF000000)

using namespace std::chrono;
namespace seio {
    
    void CrdatLoader::loadData(Tensor *X, Tensor *label, const char *rootPath, const char *datasetName, uint32 offset) {
            string binPath = string(rootPath) + "\\" + datasetName + ".crdat";
            ifstream binFile(binPath, ios::ate | ios::binary);
        
            uint64 fileSize = binFile.tellg();
            uint32 operateSize = (X->dims.size + label->dims.size * sizeof(float));
            uint32 maxOffset = fileSize / operateSize;
            assert(offset < maxOffset);
        
            //allocate the buffer if it is not allocated yet
            if((fetchBuffer) == nullptr) {
                cout<<"created fetch buffer"<<endl;
                cudaMallocHost(&fetchBuffer, operateSize);
            }
        
            //use long long to prevent overflow
            binFile.seekg((long long) offset * (long long)operateSize, ios::beg);
            binFile.read((char*)(fetchBuffer), operateSize);
            binFile.close();
        
            for(uint32 i = 0; i < X->dims.size; i++) {
                X->elements[i] = (float) (fetchBuffer[i]);
            }
        
            //fetch the labels
            cudaMemcpy(label->elements, (fetchBuffer) + X->dims.size,
                       label->dims.size * sizeof(float), cudaMemcpyHostToHost);
    }
    
    void ImgLoader::loadData(Tensor *X, Tensor *label, const char *rootPath, const char *datasetName, uint32 offset) {
        string imgPath = string(rootPath) + "\\" + datasetName;
        uint32 classID = stoi(string(datasetName).substr(0, string(datasetName).find_first_of('_')));
    
        inputImg = cv::imread(imgPath, cv::IMREAD_COLOR);
        assert(!inputImg.empty());
        
        //fetch the labels
        assert(classID < label->dims.size);
        label->elements[classID] = 1.0f;
    }
    
    void imgLoaderPPThread(ImgLoader* ldr, Tensor *X, Tensor *label){
       //bilinear resize
       float scaleH = (float)X->dims.h / (float)ldr->inputImg.rows;
       float scaleW = (float)X->dims.w / (float)ldr->inputImg.cols;
       float resizeScale = scaleH > scaleW ? scaleH : scaleW;
       unsigned char* srcDat = ldr->inputImg.data;
    
       //a, b, c, d are RGB pixel values
       //since 4 chars occupy the same memory as an int
       int a, b, c, d, x, y, index;

    }
    
    thread ImgLoader::preProcess(Tensor *X, Tensor *label) {
       return thread(imgLoaderPPThread, this, X, label);
    }
    
    void ImgLoader::prepareAsync() {
        inputImgCpy = inputImg.clone();
    }
} // seann
