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


using namespace seblas;
namespace seio {
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
        
        //pre executed before tensors write into the dataset
        DataTransformer** preProc;
        uint32 preProcStepCount;
        
        //execute with the construction of every batch and minibatch
        DataAugmentor** augmentations;
        BatchFormer* batchInitializer;
        uint32 augmentationStepCount;
        
        ProcDisplay* procDisplay;
        
        static Dataset* construct(uint32 batchSize,uint32 miniBatchSize, uint32 epochSize,
                                  uint32 allDataSize, uint32 maxEpoch, shape4 dataShape, shape4 labelShape);
        
        void setProcSteps(std::initializer_list<DataTransformer*> steps){
            preProcStepCount = steps.size();
            cudaMallocHost(&preProc, preProcStepCount * sizeof(DataTransformer*));
            for(uint32 i = 0; i < preProcStepCount; i++){
                preProc[i] = steps.begin()[i];
            }
        }
        
        void setAugmentSteps(std::initializer_list<DataAugmentor*> steps){
            augmentationStepCount = steps.size();
            cudaMallocHost(&augmentations, augmentationStepCount * sizeof(DataAugmentor*));
            for(uint32 i = 0; i < augmentationStepCount; i++){
                augmentations[i] = steps.begin()[i];
            }
        }
        
        void setProcDispaly(ProcDisplay* display){
            procDisplay = display;
        }
        
        void runPreProc();
        
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
