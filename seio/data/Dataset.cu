//
// Created by Dylan on 6/20/2022.
//

#include "Dataset.cuh"
#include "../../seblas/assist/Inspections.cuh"

namespace seio {
    
    void transformThread(int tid, int tc, Dataset* set){
        uint32 beg = tid * (set->BATCH_SIZE / tc);
        uint32 end = tid == tc - 1 ? set->BATCH_SIZE : beg + (set->BATCH_SIZE / tc);
        for(uint32 i = beg; i < end; i++){
            for(uint32 step = 0; step < set->preProcStepCount; step++){
                set->dataset[i]->X = set->preProc[step]->apply(set->dataset[i]->X);
            }
        }
    }
    
    Dataset *Dataset::construct(uint32 batchSize, uint32 miniBatchSize, uint32 epochSize,
                                       uint32 maxEpoch, shape4 dataShape, shape4 labelShape) {
        assert(batchSize > 0);
        assert(epochSize > 0);
        assert(batchSize <= epochSize);
    
        Dataset* out;
        cudaMallocHost(&out, sizeof(Dataset));
        out->BATCH_SIZE = batchSize;
        out->EPOCH_SIZE = epochSize;
        out->MAX_EPOCH = maxEpoch;
        out->MINI_BATCH_SIZE = miniBatchSize;
    
        assert(out->BATCH_SIZE % out->MINI_BATCH_SIZE == 0);
        
        out->enableDynamicLoading = true;
        
        cudaMallocHost(&out->dataBatch[0], (batchSize/miniBatchSize) * sizeof(Data*));
        cudaMallocHost(&out->dataBatch[1], (batchSize/miniBatchSize) * sizeof(Data*));
        cudaMallocHost(&out->dataset, (batchSize) * sizeof(Data*));
        
        shape4 procDataShape = {miniBatchSize, dataShape.c, dataShape.h, dataShape.w};
        shape4 procLabelShape = {miniBatchSize, labelShape.c, labelShape.h, labelShape.w};
        
        for(uint32 i = 0; i < batchSize/miniBatchSize; i++) {
            out->dataBatch[0][i] = Data::declare(procDataShape, procLabelShape)->instantiate();
            out->dataBatch[1][i] = Data::declare(procDataShape, procLabelShape)->instantiate();
        }
        
        //only one batch of data would be stored in our host memory
        for(uint32 i = 0; i < batchSize; i++) {
            out->dataset[i] = Data::declare(dataShape, labelShape)->instantiateHost();
        }
    
        out->dataShape = dataShape;
        out->labelShape = labelShape;
        
        out->batchInitializer = new BatchFormer(miniBatchSize, dataShape, labelShape);
    
        assertCuda(__FILE__, __LINE__);
        return out;
    }
    
    void Dataset::genBatch() {
        fetcher->fetchTrain(dataset, BATCH_SIZE);
        runPreProc();
        for(int i = 0; i < BATCH_SIZE / MINI_BATCH_SIZE; i++) {
            //Run augmentation steps
            Data* src = batchInitializer->form(dataset, i);
            for(uint32 step = 0; step < augmentationStepCount; step++){
                src = augmentations[step]->apply(src);
            }
        
            cudaMemcpy(dataBatch[batchID%2][i]->X->elements, src->X->elements,
                       sizeof(float) * src->X->dims.size, cudaMemcpyHostToDevice);
            cudaMemcpy(dataBatch[batchID%2][i]->label->elements, src->label->elements,
                       sizeof(float) * src->label->dims.size, cudaMemcpyHostToDevice);
            assertCuda(__FILE__, __LINE__);
        }
        
        //change batchID
        batchID++;
        epochID = (batchID * BATCH_SIZE) / EPOCH_SIZE;
    }
    
    void genBatchThread(Dataset* set) {
        set->genBatch();
    }
    
    //this method uses an async way of generating the next batch data
    //while training is running on the current batch
    thread Dataset::genBatchAsync() {
        return thread(genBatchThread, this);
    }
    
    void Dataset::allocTestSet() {
        uint32 testSetSize = fetcher->TEST_SIZE;
        assert(testSetSize % MINI_BATCH_SIZE == 0);
        cudaMallocHost(&testset, testSetSize * sizeof(Data*));
        shape4 procDataShape = {MINI_BATCH_SIZE, dataShape.c, dataShape.h, dataShape.w};
        shape4 procLabelShape = {MINI_BATCH_SIZE, labelShape.c, labelShape.h, labelShape.w};
        TEST_SIZE = testSetSize;
    
        for(uint32 i = 0; i < testSetSize; i++) {
            testset[i] = Data::declare(procDataShape, procLabelShape)->instantiate();
        }
        
        auto* dataBuf = Tensor::createHost(dataShape);
        auto* labelBuf = Tensor::createHost(labelShape);
        
        for(uint32 i = 0; i < testSetSize / MINI_BATCH_SIZE; i++){
            for (int proc = 0; proc < MINI_BATCH_SIZE; proc++) {
                fetcher->fetchTest(dataBuf, labelBuf);
                
                //Run load-time steps
                for(uint32 step = 0; step < preProcStepCount; step++){
                    dataBuf = preProc[step]->apply(dataBuf);
                }
                
                //create mini-batch
                cudaMemcpy(testset[i]->X->elements + dataShape.size * proc,
                           dataBuf->elements,
                           dataShape.size * sizeof(float),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(testset[i]->label->elements + labelShape.size * proc,
                            labelBuf->elements,
                            labelShape.size * sizeof(float),
                            cudaMemcpyHostToDevice);
                assertCuda(__FILE__, __LINE__);
            }
        }
        dataBuf->eliminateHost();
        labelBuf->eliminateHost();
    }
    
    void Dataset::runPreProc() {
        _alloc<CPU_THREADS>(transformThread, this);
    }
    
    void Dataset::setProcSteps(std::initializer_list<DataTransformer *> steps) {
        preProcStepCount = steps.size();
        cudaMallocHost(&preProc, preProcStepCount * sizeof(DataTransformer*));
        for(uint32 i = 0; i < preProcStepCount; i++){
            preProc[i] = steps.begin()[i];
        }
    }
    
    void Dataset::setAugmentSteps(std::initializer_list<DataAugmentor *> steps) {
        augmentationStepCount = steps.size();
        cudaMallocHost(&augmentations, augmentationStepCount * sizeof(DataAugmentor*));
        for(uint32 i = 0; i < augmentationStepCount; i++){
            augmentations[i] = steps.begin()[i];
        }
    }
    
    void Dataset::setDataFetcher(DataFetcher *fetch) {
        this->fetcher = fetch;
    }
} // seann