//
// Created by Dylan on 6/20/2022.
//

#include "Dataset.cuh"
#include "../../seblas/assist/Inspections.cuh"

namespace seio {
    
    void transformThread(int tid, int tc, Dataset* set){
        uint32 beg = tid * (set->EPOCH_SIZE / tc);
        uint32 end = tid == tc - 1 ? set->EPOCH_SIZE : beg + (set->EPOCH_SIZE / tc);
        for(uint32 i = beg; i < end; i++){
            for(uint32 step = 0; step < set->preProcStepCount; step++){
                set->dataset[i]->X = set->preProc[step]->apply(set->dataset[i]->X);
            }
        }
    }
    
    Dataset* Dataset::construct(uint32 batchSize, uint32 miniBatchSize, uint32 epochSize, uint32 allDataSize,
                                uint32 maxEpoch,shape4 dataShape, shape4 labelShape) {
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
        
        cudaMallocHost(&out->dataBatch[0], (batchSize/miniBatchSize) * sizeof(Data*));
        cudaMallocHost(&out->dataBatch[1], (batchSize/miniBatchSize) * sizeof(Data*));
        
        cudaMallocHost(&out->dataset, allDataSize * sizeof(Data*));
        
        shape4 procDataShape = {miniBatchSize, dataShape.c, dataShape.h, dataShape.w};
        shape4 procLabelShape = {miniBatchSize, labelShape.c, labelShape.h, labelShape.w};
        
        for(uint32 i = 0; i < batchSize/miniBatchSize; i++) {
            out->dataBatch[0][i] = Data::declare(procDataShape, procLabelShape)->instantiate();
            out->dataBatch[1][i] = Data::declare(procDataShape, procLabelShape)->instantiate();
        }
        
        for(uint32 i = 0; i < allDataSize; i++){
            out->dataset[i] = Data::declare(dataShape, labelShape);
        }
        
        out->dataShape = dataShape;
        out->labelShape = labelShape;
        
        out->remainedData = out->EPOCH_SIZE/out->MINI_BATCH_SIZE;
        out->batchInitializer = new BatchFormer(miniBatchSize, dataShape, labelShape);
        
        assertCuda(__FILE__, __LINE__);
        
        return out;
    }
    
    inline void shift(Data** dataset, uint32 size, uint32 index, uint32 step){
        for(int proc = 0; proc < step; proc++){
            Data* selected = dataset[index];
            for(uint32 i = index; i < size - 1; i++){
                dataset[i] = dataset[i + 1];
            }
            dataset[size - 1] = selected;
        }
    }
    
    void Dataset::genBatch() {
        auto p1 = std::chrono::system_clock::now();
        default_random_engine generator(
                chrono::duration_cast<std::chrono::microseconds>(
                        p1.time_since_epoch()).count()
        );
        
        for(int i = 0; i < BATCH_SIZE / MINI_BATCH_SIZE; i++) {
            
            //find the current epoch parameters
            uniform_int_distribution<uint32> distribution(0, remainedData-1);
            uint32 index = distribution(generator);
            
            //Run augmentation steps
            Data* src = batchInitializer->form(dataset, index);
            for(uint32 step = 0; step < augmentationStepCount; step++){
                src = augmentations[step]->apply(src);
            }
            
            cudaMemcpy(dataBatch[batchID%2][i]->X->elements, src->X->elements,
                       sizeof(float) * src->X->dims.size, cudaMemcpyHostToDevice);
            cudaMemcpy(dataBatch[batchID%2][i]->label->elements, src->label->elements,
                          sizeof(float) * src->label->dims.size, cudaMemcpyHostToDevice);
            assertCuda(__FILE__, __LINE__);
            
            //change data location to prevent repeating use of same data
            shift(dataset, EPOCH_SIZE,  index * MINI_BATCH_SIZE, MINI_BATCH_SIZE);
            remainedData --;
            
            if(remainedData == 0){
                epochID++;
                remainedData = EPOCH_SIZE / MINI_BATCH_SIZE;
            }
        }
        //change batchID
        batchID++;
    }
    
    
    
    void genBatchThread(Dataset* set) {
        set->genBatch();
    }
    
    //this method uses an async way of generating the next batch data
    //while training is running on the current batch
    thread Dataset::genBatchAsync() {
        return thread(genBatchThread, this);
    }
    
    void Dataset::allocTestSet(uint32 testSetSize) {
        assert(testSetSize % MINI_BATCH_SIZE == 0);
        cudaMallocHost(&testset, testSetSize * sizeof(Data*));
        shape4 procDataShape = {MINI_BATCH_SIZE, dataShape.c, dataShape.h, dataShape.w};
        shape4 procLabelShape = {MINI_BATCH_SIZE, labelShape.c, labelShape.h, labelShape.w};
        TEST_SIZE = testSetSize;
        
        for(uint32 i = 0; i < testSetSize / MINI_BATCH_SIZE; i++){
            //directly alloc the testset onto the GPU
            testset[i] = Data::declare(procDataShape, procLabelShape)->instantiate();
            
            //copy data to CUDA memory
            for (int proc = 0; proc < MINI_BATCH_SIZE; proc++) {
                cudaMemcpy(testset[i]->X->elements + dataShape.size * proc,
                           dataset[EPOCH_SIZE + i * MINI_BATCH_SIZE + proc]->X->elements,
                           dataShape.size * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
            assertCuda(__FILE__, __LINE__);
    
            for (int proc = 0; proc < MINI_BATCH_SIZE; proc++) {
                cudaMemcpy(testset[i]->label->elements + labelShape.size * proc,
                           dataset[EPOCH_SIZE + i * MINI_BATCH_SIZE + proc]->label->elements,
                           labelShape.size * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
            assertCuda(__FILE__, __LINE__);
        }
    }
    
    void Dataset::runPreProc() {
        _alloc<CPU_THREADS>(transformThread, this);
    }
} // seann