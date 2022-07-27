//
// Created by Dylan on 6/20/2022.
//

#include "DataLoader.cuh"
#include "../../seblas/assist/Inspections.cuh"
#include <sys/stat.h>


#define toLittleEndian(i) ((i>>24)&0xFF) | ((i>>8)&0xFF00) | ((i<<8)&0xFF0000) | ((i<<24)&0xFF000000)

namespace seio {
    
    void readBytes(unsigned char *buffer, unsigned long size, const char *binPath) {
        FILE *fp = fopen(binPath, "rb");
        assert(fp != nullptr);
        fread(buffer, sizeof(unsigned char), size, fp);
        fclose(fp);
    }
    
    unsigned long getFileSize(const char *binPath) {
        std::ifstream in(binPath, std::ifstream::ate | std::ifstream::binary);
        unsigned long size = in.tellg();
        in.close();
        return size;
    }
    
    Tensor *fetchOneHotLabel(const unsigned char *buffer, uint32 offset, shape4 dims) {
        Tensor *label = Tensor::createHost(dims);
        unsigned char labelVal = buffer[offset];
        for (uint32 i = 0; i < label->dims.size; i++) {
            label->elements[i] = (labelVal == i) ? 1.0f : 0.0f;
        }
        return label;
    }
    
    //The IDX data must be in NCHW arrangement
    Tensor *fetchBinImage(const unsigned char *buffer, uint32 offset, shape4 dims) {
        Tensor *image = Tensor::createHost(dims);
        for (uint32 i = 0; i < image->dims.size; i++) {
            image->elements[i] = (float) buffer[offset + i];
        }
        return image;
    }
    
    void fetchCrdat(Tensor* x, Tensor* label, const char* rootPath, const char* datasetName,
                    uint32 offset, unsigned char** buffer) {
        string binPath = string(rootPath) + "\\" + datasetName + ".crdat";
        ifstream binFile(binPath, ios::ate | ios::binary);
        
        uint64 fileSize = binFile.tellg();
        uint32 operateSize = (x->dims.size + label->dims.size * sizeof(float));
        uint32 maxOffset = fileSize / operateSize;
        assert(offset < maxOffset);
        
        //allocate the buffer if it is not allocated yet
        if((*buffer) == nullptr) {
            cout<<"created fetch buffer"<<endl;
            cudaMallocHost(buffer, operateSize);
        }
        
        //use long long to prevent overflow
        binFile.seekg((long long) offset * (long long)operateSize, ios::beg);
        binFile.read((char*)(*buffer), operateSize);
        binFile.close();
    
        for(uint32 i = 0; i < x->dims.size; i++) {
            x->elements[i] = (float) (*buffer)[i];
        }
        
        //fetch the labels
        cudaMemcpy(label->elements, (*buffer) + x->dims.size,
                   label->dims.size * sizeof(float), cudaMemcpyHostToHost);
    }
} // seann