//
// Created by Dylan on 7/24/2022.
//

#include "DataFetcher.cuh"
#include "../../seblas/assist/Inspections.cuh"

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
        thread t;
        for(uint32 i = 0; i < step; i++){
            uniform_int_distribution<uint32> distribution(0, remainedData - 1);
            uint32 index = distribution(generator);
            uint32 loadIndex = registry[index].offsetID;
            //wait for previous image to stop async loading
            if(i>0) t.join();
            if(registry[index].WITH_LOCATION){
                dataLoader->loadData(dataset[i]->X, dataset[i]->label, ROOT_PATH.c_str(),
                                     registry[index].elementName, loadIndex);
                dataLoader->prepareAsync();
                t = dataLoader->preProcess(dataset[i]->X, dataset[i]->label);
            }else {
                dataLoader->loadData(dataset[i]->X, dataset[i]->label, ROOT_PATH.c_str(),
                                     DATASET_NAME.c_str(), loadIndex);
            }
            shift(registry, remainedData, index);
            remainedData--;
            
            if(remainedData <= 0){
                remainedData = EPOCH_SIZE;
            }
        }
        t.join();
    }
    
    void DataFetcher::fetchTest(Tensor * data, Tensor * label) {
        auto p1 = std::chrono::system_clock::now();
        default_random_engine generator(
                chrono::duration_cast<std::chrono::microseconds>(
                        p1.time_since_epoch()).count()
        );
    
        uniform_int_distribution<uint32> distribution(0, remainedData - 1);
        uint32 index = distribution(generator);
        uint32 loadIndex = registry[index].offsetID;
        
        //when dealing with images, we would need to get the corresponding path
        if(registry[index].WITH_LOCATION){
            dataLoader->loadData(data, label, ROOT_PATH.c_str(),
                                 registry[index].elementName, loadIndex);
            dataLoader->prepareAsync();
            dataLoader->preProcess(data, label).join();
        }else {
            dataLoader->loadData(data, label, ROOT_PATH.c_str(),
                                 DATASET_NAME.c_str(), loadIndex);
        }
    }
} // seio