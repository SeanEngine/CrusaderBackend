//
// Created by Dylan on 6/20/2022.
//

#include "Sequential.cuh"
#include "../../seblas/assist/Inspections.cuh"
#include "../../seio/modelcache/codec.cuh"
#define MODEL_CACHE_PATH "D:\\Projects\\CLionProjects\\Crusader\\ModelSav\\"

namespace seann {
    void Sequential::waive() const {
        for (int i = 1; i < OPERAND_COUNT; i++) {
            operands[i]->bindPrev(operands[i - 1]);
            operands[i]->bindInput(operands[i - 1]->Y);
        }
        
        for (int i = 0; i < OPERAND_COUNT - 1; i++) {
            operands[i]->bindNext(operands[i + 1]);
        }
    }
    
    void Sequential::construct(OptimizerInfo *info) {
        logInfo(seio::LOG_SEG_SEANN, "Constructing Model: ");
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->initNetParams(info, i == 0 ? netX->A->dims : operands[i - 1]->Y->A->dims);
            operands[i]->operandID = i;
            logInfo(seio::LOG_SEG_SEANN, operands[i]->info());
        }
        
        //bind inputs and outputs
        operands[0]->X->inherit(netX);
        netY = Parameter::declare(operands[OPERAND_COUNT - 1]->Y->A->dims)
                ->inherit(operands[OPERAND_COUNT - 1]->Y);
        
        //bind the layers together
        waive();
        
        for(int i = 0; i < OPERAND_COUNT; i++){
            operands[i]->postWaiveInit(info);
        }
    }
    
    Tensor *Sequential::forward() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->forward();
        }
        return netY->A;
    }
    
    Tensor *Sequential::forward(Tensor *X) const {
        X->copyToD2D(netX->A);
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->forward();
        }
        return netY->A;
    }
    
    Tensor *Sequential::inferenceForward() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->inferenceForward();
        }
        return netY->A;
    }
    
    Tensor *Sequential::inferenceForward(Tensor *X) const {
        X->copyToD2D(netX->A);
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->inferenceForward();
        }
        return netY->A;
    }
    
    void Sequential::randInit() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->randFillNetParams();
        }
    }
    
    Tensor *Sequential::backward(Tensor *label) const {
        loss(netY, label);
        for (int i = (int) OPERAND_COUNT - 1; i >= 0; i--) {
            operands[i]->xGrads();
            operands[i]->paramGrads();
        }
        return netX->dA;
    }
    
    void Sequential::setLoss(LossFunc lossFunc, LossFuncCalc lossFWD) {
        loss = lossFunc;
        lossFW = lossFWD;
    }
    
    void Sequential::learn() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->updateParams();
        }
    }
    
    void Sequential::learnBatch() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->batchUpdateParams();
        }
    }

//this train method does not support BN
    void Sequential::train(Dataset *data, bool WITH_TEST, uint32 TEST_FREQUENCY) {
        assert(data->BATCH_SIZE > 0 && data->BATCH_SIZE % netX->A->dims.n == 0);
        data->genBatch();
        Tensor* lossBuf = Tensor::create(netY->A->dims);
        
        remove(MODEL_CACHE_PATH "cache.log");
        fstream trainLog = fstream(MODEL_CACHE_PATH "cache.log", ios::binary | ios::out);
        data->procDisplay->show(0);
        while (data->epochID < data->MAX_EPOCH) {
            uint32 batchID = data->batchID - 1;
            uint32 epochID = data->epochID;
            auto pass = data->genBatchAsync();
            float batchLoss = 0;
            
            //training over each sample in the batch
            for (uint32 sampleID = 0; sampleID < data->BATCH_SIZE / data->MINI_BATCH_SIZE; sampleID ++) {
                forward(data->dataBatch[batchID % 2][sampleID]->X);
                backward(data->dataBatch[batchID % 2][sampleID]->label);
                
                float lossVal = lossFW(netY, data->dataBatch[batchID % 2][sampleID]->label, lossBuf);
                batchLoss += lossVal;
                learn();
            }
            
            trainLog << batchLoss / (float) data->BATCH_SIZE << ", ";
            
            //BGD updates
            learnBatch();
            pass.join(); //wait for next batch to prefetch
    
            float lossVal = 0;
            if(WITH_TEST){
                if(data->batchID % TEST_FREQUENCY == 0){
                    for(int i = 0; i < data->TEST_SIZE/data->MINI_BATCH_SIZE; i++){
                        inferenceForward(data->testset[i]->X);
                        //inspect(netY->A);
                        lossVal += lossFW(netY, data->testset[i]->label, lossBuf);
                    }
                    lossVal /= (float) data->TEST_SIZE;
                    trainLog << lossVal << ",";
                }
            }
    
            if (data->epochID != epochID) {
                for (int i = 0; i < OPERAND_COUNT; i++) {
                    operands[i]->zeroGrads();
                }
                data->procDisplay->reset();
                if(epochID % 5 == 0){
                    remove(MODEL_CACHE_PATH "cache.crseq");
                    saveSequence(MODEL_CACHE_PATH "cache.crseq", this, epochID);
                }
                data->procDisplay->show(data->epochID);
            }
    
            data->procDisplay->update(batchLoss / (float) data->BATCH_SIZE , lossVal);
        }
        trainLog.close();
    }
}