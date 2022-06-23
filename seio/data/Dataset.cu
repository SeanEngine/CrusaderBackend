//
// Created by Dylan on 6/20/2022.
//

#include "Dataset.cuh"

namespace seio {
    Data *Data::declare(shape4 dataShape, shape4 labelShape) {
        Data* out;
        cudaMallocHost(&out, sizeof(Data));
        out->X = Tensor::declare(dataShape);
        out->label = Tensor::declare(labelShape);
        return out;
    }
    
    Data *Data::instantiate() {
        X->instantiate();
        label->instantiate();
        return this;
    }
    
    Data *Data::instantiateHost() {
        X->instantiateHost();
        label->instantiateHost();
        return this;
    }
    
    Data *Data::inherit(Tensor *X0, Tensor *label0) {
        X->attach(X0);
        label->attach(label0);
        return this;
    }
    
    Data *Data::copyOffD2D(Data *onDevice) {
        onDevice->X->copyToD2D(X);
        onDevice->label->copyToD2D(label);
        return this;
    }
    
    Data *Data::copyOffH2D(Data *onHost) {
        onHost->X->copyToH2D(X);
        onHost->label->copyToH2D(label);
        return this;
    }
    
    Data *Data::copyOffD2H(Data *onDevice) {
        onDevice->X->copyToD2H(X);
        onDevice->label->copyToD2H(label);
        return this;
    }
    
    Data *Data::copyOffH2H(Data *onHost) {
        onHost->X->copyToH2H(X);
        onHost->label->copyToH2H(label);
        return this;
    }
    
    Data *Data::copyOffD2D(Tensor *X0, Tensor *label0) {
        X0->copyToD2D(X);
        label0->copyToD2D(label);
        return this;
    }
    
    Data *Data::copyOffH2D(Tensor *X0, Tensor *label0) {
        X0->copyToH2D(X);
        label0->copyToH2D(label);
        return this;
    }
    
    Data *Data::copyOffD2H(Tensor *X0, Tensor *label0) {
        X0->copyToD2H(X);
        label0->copyToD2H(label);
        return this;
    }
    
    Data *Data::copyOffH2H(Tensor *X0, Tensor *label0) {
        X0->copyToH2H(X);
        label0->copyToH2H(label);
        return this;
    }
    
    void Data::destroy() {
        X->destroy();
        label->destroy();
        cudaFree(this);
    }
    
    void Data::destroyHost() {
        X->destroyHost();
        label->destroyHost();
        cudaFreeHost(this);
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
            
            //copy data to CUDA memory
            for (int proc = 0; proc < MINI_BATCH_SIZE; proc++) {
                cudaMemcpy(dataBatch[batchID%2][i]->X->elements + dataShape.size * proc,
                           dataset[index * MINI_BATCH_SIZE + proc]->X->elements,
                           dataShape.size * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
            assertCuda(__FILE__, __LINE__);
    
            for (int proc = 0; proc < MINI_BATCH_SIZE; proc++) {
                cudaMemcpy(dataBatch[batchID%2][i]->label->elements + labelShape.size * proc,
                           dataset[index * MINI_BATCH_SIZE + proc]->label->elements,
                           labelShape.size * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
            assertCuda(__FILE__, __LINE__);
            
            //change data location to prevent repeating use of same data
            shift(dataset, EPOCH_SIZE, index * MINI_BATCH_SIZE, MINI_BATCH_SIZE);
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
} // seann