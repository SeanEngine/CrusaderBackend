//
// Created by Dylan on 6/20/2022.
//

#include "DataLoader.cuh"

#define toLittleEndian(i) ((i>>24)&0xFF) | ((i>>8)&0xFF00) | ((i<<8)&0xFF0000) | ((i<<24)&0xFF000000)


namespace seio {
    
    void readBytes(BYTE *buffer, unsigned long size, const char* binPath){
        FILE *fp = fopen(binPath, "rb");
        assert(fp != nullptr);
        fread(buffer, sizeof(BYTE), size, fp);
        fclose(fp);
    }
    
    unsigned long getFileSize(const char* binPath){
        std::ifstream in(binPath, std::ifstream::ate | std::ifstream::binary);
        unsigned long size = in.tellg();
        in.close();
        logDebug(LOG_SEG_SEIO,"Preparing file \"" + string(binPath) + "\" : size = "
                              + to_string(size), LOG_COLOR_LIGHT_YELLOW);
        return size;
    }
    
    Tensor* fetchOneHotLabel(const BYTE* buffer, uint32 offset, shape4 dims){
        Tensor* label = Tensor::createHost(dims);
        BYTE labelVal = buffer[offset];
        for(uint32 i = 0; i < label->dims.size; i++){
            label->elements[i] = (labelVal == i) ? 1.0f : 0.0f;
        }
        return label;
    }
    
    //The IDX data must be in NCHW arrangement
    Tensor* fetchBinImage(const BYTE* buffer, uint32 offset, shape4 dims){
        Tensor* image = Tensor::createHost(dims);
        for(uint32 i = 0; i < image->dims.size; i++){
            image->elements[i] = (float)buffer[offset + i];
        }
        return image;
    }
    
    void fetchIDXThread(int tid, int tc, Tensor*(*decode)(const BYTE*, uint32, shape4),
                        Dataset* set, BYTE* buf,  uint32 begOffset, uint32 step, bool isLabel){
        int start = tid * (int)(set->EPOCH_SIZE / tc);
        int end = tid == tc - 1 ? (int)set->EPOCH_SIZE : start + (int)(set->EPOCH_SIZE / tc);
        
        for(int i = start; i < end; i++){
            uint32 offset = begOffset + i * step;
            if(isLabel){
                set->dataset[i]->label = decode(buf, offset, set->labelShape);
            }else{
                set->dataset[i]->X = decode(buf, offset, set->dataShape);
            }
        }
    }
    
    Dataset* fetchIDX(Dataset* dataset, const char* binPath, uint32 step, bool isLabel){
        
        BYTE* buffer;
        unsigned long size = getFileSize(binPath);
        cudaMallocHost(&buffer, size);
        
        //load IDX format headers
        readBytes(buffer, size, binPath);
        uint32 magic = toLittleEndian(*(uint32*)buffer);
        uint32 numItems = toLittleEndian(*(uint32*)(buffer + 4));
        
        logDebug(LOG_SEG_SEIO,"Loading IDX file : magic = "
                              + to_string(magic) + ", numItems = " + to_string(numItems));
        assert(numItems == (int)dataset->EPOCH_SIZE);
        
        //load dimension info:
        uint32 dimCount = magic & 0xFF;
        uint32 START_OFFSET = isLabel ? 8 : dimCount * 4 + 8;
        
        _alloc<CPU_THREADS>(fetchIDXThread, isLabel ? fetchOneHotLabel : fetchBinImage,
                            dataset, buffer, START_OFFSET, step, isLabel);
        logDebug(LOG_SEG_SEIO, "Fetched IDX file for : " + to_string(dataset->EPOCH_SIZE) + " items");
        
        cudaFreeHost(buffer);
        return dataset;
    }
    
    void fetchCIFARThread(int tid, int tc, Dataset* set, BYTE* buf, uint32 fileID, uint32 fileItems){
        int start = tid * (int)(fileItems / tc);
        int end = tid == tc - 1 ? (int)fileItems : start + (int)(fileItems / tc);
        
        //CIFAR Dataset keeps files with images and its labels together
        uint32 step = set->dataShape.size +  1;
        
        for(uint32 proc = start; proc < end; proc++){
            uint32 offset = step * proc;
            auto* lab = fetchOneHotLabel(buf, offset, set->labelShape);
            auto* dat = fetchBinImage(buf, offset + 1, set->dataShape);
            set->dataset[fileID * fileItems + proc]->X->elements = dat->elements;
            set->dataset[fileID * fileItems + proc]->label->elements = lab->elements;
        }
    }
    
    Dataset* fetchCIFAR(Dataset* dataset, const char* binPath, uint32 fileID){
        BYTE* buffer;
        unsigned long size = getFileSize(binPath);
        cudaMallocHost(&buffer, size);
        readBytes(buffer, size, binPath);
        
        //all files should be of the same size for CIFAR-10
        uint32 fileItems = size / (dataset->dataShape.size + 1);
        logDebug(LOG_SEG_SEIO,"Loading CIFAR file : numItems = " + to_string(fileItems));
        
        _alloc<1>(fetchCIFARThread, dataset, buffer, fileID, fileItems);
        logDebug(LOG_SEG_SEIO, "Fetched IDX file for : " + to_string(fileItems) + " items");
        return dataset;
    }
} // seann