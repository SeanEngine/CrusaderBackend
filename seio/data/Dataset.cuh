//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_DATASET_CUH
#define CRUSADER_DATASET_CUH

#include "../../seblas/operations/cuOperations.cuh"

using namespace seblas;
namespace seio {
    struct Data{
    public:
        Tensor* X;
        Tensor* label;
        
        static Data* declare(shape4 dataShape, shape4 labelShape);
        
        Data* instantiate();
        
        Data* instantiateHost();
        
        void destroy();
        
        void destroyHost();
        
        Data* inherit(Tensor* X0, Tensor* label0);
        
        Data* copyOffD2D(Data* onDevice);
        
        Data* copyOffH2D(Data* onHost);
        
        Data* copyOffD2H(Data* onDevice);
        
        Data* copyOffH2H(Data* onHost);
        
        Data* copyOffD2D(Tensor* X0, Tensor* label0);
        
        Data* copyOffH2D(Tensor* X0, Tensor* label0);
        
        Data* copyOffD2H(Tensor* X0, Tensor* label0);
        
        Data* copyOffH2H(Tensor* X0, Tensor* label0);
    };
    
    struct Dataset {
    public:
        //onDevice
        Data** dataBatch[2];
        
        //onHost : supports epoch operations
        Data** dataset;
        Data** testset;
        
        uint32 BATCH_SIZE;
        uint32 EPOCH_SIZE;
        uint32 MAX_EPOCH;
        uint32 MINI_BATCH_SIZE;
        uint32 TEST_SIZE;
        
        uint32 batchID = 0;
        uint32 epochID = 0;
        uint32 remainedData;
        
        shape4 dataShape;
        shape4 labelShape;
        
        static Dataset* construct(uint32 batchSize,uint32 miniBatchSize, uint32 epochSize,
                                  uint32 allDataSize, uint32 maxEpoch, shape4 dataShape, shape4 labelShape);
        
        //The test set will be the last few instances of the input dataset
        void allocTestSet(uint32 testSetSize);
        
        //generate a data batch
        void genBatch();
        
        //this method uses an async way of generating the next batch data
        //while training is running on the current batch
        thread genBatchAsync();
    };
    
} // seann

#endif //CRUSADER_DATASET_CUH
