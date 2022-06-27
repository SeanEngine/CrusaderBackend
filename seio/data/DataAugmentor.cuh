//
// Created by Dylan on 6/27/2022.
//

#ifndef CRUSADER_DATAAUGMENTOR_CUH
#define CRUSADER_DATAAUGMENTOR_CUH

#include "../../seblas/tensor/Tensor.cuh"
#include "Data.cuh"

using namespace seblas;

namespace seio{
    
    //group individual data together into mini batches (A host function)
    void batchInit(Data** set, uint32 begin, uint32 miniBatchSize, Data* buf);
    
    class DataAugmentor{
    public:
        uint32 miniBatchSize;
        shape4 dataShape;
        shape4 labelShape;
        Data* buf;
        
        DataAugmentor(uint32 miniBatchSize, shape4 dataShape, shape4 labelShape) :
        miniBatchSize(miniBatchSize), dataShape(dataShape), labelShape(labelShape){
            this->buf = Data::declare(
                    {miniBatchSize, dataShape.c, dataShape.h, dataShape.w},
                    {miniBatchSize, labelShape.c, labelShape.h, labelShape.w}
                    )->instantiateHost();
        }
        
        virtual Data* apply(Data* src) = 0;
    };
    
    class BatchFormer : public DataAugmentor{
    public:
        BatchFormer(uint32 miniBatchSize, shape4 dataShape, shape4 labelShape) :
        DataAugmentor(miniBatchSize, dataShape, labelShape){}
        
        Data* apply(Data* src) override{return src;}
        
        Data* form(Data** set, uint32 begin){
            batchInit(set, begin, miniBatchSize, buf);
            return buf;
        }
    };
    
    class RandCorp : public DataAugmentor{
    public:
        uint32 padH;
        uint32 padW;
        
        RandCorp(uint32 miniBatchSize, shape4 dataShape, shape4 labelShape
                 , uint32 padH, uint32 padW) :
                DataAugmentor(miniBatchSize, dataShape, labelShape)
        , padH(padH), padW(padW){}
    
        Data* apply(Data* src) override;
    };
    
    class RandFlipW : public DataAugmentor{
    public:
        RandFlipW(uint32 miniBatchSize, shape4 dataShape, shape4 labelShape) :
                DataAugmentor(miniBatchSize, dataShape, labelShape){}
    
        Data* apply(Data* src) override;
    };
}


#endif //CRUSADER_DATAAUGMENTOR_CUH
