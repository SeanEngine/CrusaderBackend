//
// Created by Dylan on 7/24/2022.
//

#include "DataFetcher.cuh"

namespace seio {
    
    inline void shift(DataPoint* dataset, uint32 size, uint32 index){
        DataPoint selected = dataset[index];
        for(uint32 i = index; i < size - 1; i++){
            dataset[i] = dataset[i + 1];
        }
        dataset[size - 1] = selected;
    }
    
    void DataFetcher::fetchTrain(Data** dataset, uint32 step) {
        auto p1 = std::chrono::system_clock::now();
        default_random_engine generator(
                chrono::duration_cast<std::chrono::microseconds>(
                        p1.time_since_epoch()).count()
        );
        
        for(uint32 i = 0; i < step; i++){
            uniform_int_distribution<uint32> distribution(0, remainedData - 1);
            uint32 index = distribution(generator);
            uint32 loadIndex = registry[index].offsetID;
            fetch(dataset[i]->X, dataset[i]->label, ROOT_PATH.c_str(), DATASET_NAME.c_str(), loadIndex, fetchBuffer);
            shift(registry, remainedData, index);
            remainedData--;
            
            if(remainedData == 0){
                remainedData = EPOCH_SIZE;
            }
        }
    }
    
    void DataFetcher::fetchTest(Tensor * data, Tensor * label, uint32 testID) {
        fetch(data, label, ROOT_PATH.c_str(), DATASET_NAME.c_str(), EPOCH_SIZE + testID, fetchBuffer);
    }
} // seio