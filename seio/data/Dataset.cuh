//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_DATASET_CUH
#define CRUSADER_DATASET_CUH

#include "../../seblas/operations/cuOperations.cuh"
#include "DataAugmentor.cuh"
#include "DataTransform.cuh"
#include "Data.cuh"
#include "../watchDog/TrainProcDisplay.cuh"
#include "DataFetcher.cuh"


using namespace seblas;
namespace seio {
    struct Dataset {
    public:
        //onDevice
        Data** dataBatch[2];
        
        //onHost : supports epoch operations
        Data** dataset;
        Data** testset;
        
        //dynamic fatching parameters (if enabled)
        DataFetcher* fetcher;
        
        //standard dataset parameters
        uint32 BATCH_SIZE;
        uint32 EPOCH_SIZE;
        uint32 MAX_EPOCH;
        uint32 MINI_BATCH_SIZE;
        uint32 TEST_SIZE;
        
        uint32 batchID = 0;
        uint32 epochID = 0;
        
        //this is used when host memory is limited
        uint32 dynamicLoadSize;
        bool enableDynamicLoading;
        
        shape4 dataShape;
        shape4 labelShape;
        
        //pre executed before tensors write into the dataset
        DataTransformer** preProc;
        uint32 preProcStepCount;
        
        //execute with the construction of every batch and minibatch
        DataAugmentor** augmentations;
        BatchFormer* batchInitializer;
        uint32 augmentationStepCount;
        
        ProcDisplay* procDisplay;
        
        //used when you have no idea about the actual size of your dataset
        static Dataset* construct(uint32 batchSize, uint32 miniBatchSize,uint32 epochSize,
                                         uint32 maxEpoch, shape4 dataShape, shape4 labelShape);
        
        void setDataFetcher(DataFetcher* fetcher);
        
        void setProcSteps(std::initializer_list<DataTransformer*> steps);
        
        void setAugmentSteps(std::initializer_list<DataAugmentor*> steps);
        
        void setProcDispaly(ProcDisplay* display){
            procDisplay = display;
        }
        
        void runPreProc();
        
        //The test set will be the last few instances of the input dataset
        void allocTestSet();
        
        //generate a data batch
        void genBatch();
        
        //this method uses an async way of generating the next batch data
        //while training is running on the current batch
        thread genBatchAsync();
    };
    
} // seann

#endif //CRUSADER_DATASET_CUH
