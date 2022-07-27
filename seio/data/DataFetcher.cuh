//
// Created by Dylan on 7/24/2022.
//

#ifndef CRUSADER_DATAFETCHER_CUH
#define CRUSADER_DATAFETCHER_CUH

#include <utility>

#include "Data.cuh"

namespace seio {
    //[Data] [Label] [Root Path] [Dataset Name] [load offset]
    typedef void (*DynamicFetchFunc)(Tensor*, Tensor*, const char*, const char*, uint32, unsigned char**);
    
    struct DataPoint {
        uint32 offsetID;
    };
    
    class DataFetcher {
    public:
        string ROOT_PATH;
        string DATASET_NAME;
        uint32 EPOCH_SIZE;
        DynamicFetchFunc fetch;
        uint32 remainedData;
        DataPoint* registry{};
        
        unsigned char* fetchBuffer = nullptr;
        
        DataFetcher(string rootPath, string datasetName, uint32 EPOCH_SIZE, DynamicFetchFunc fetchFunc) {
            ROOT_PATH = std::move(rootPath);
            DATASET_NAME = std::move(datasetName);
            this->EPOCH_SIZE = EPOCH_SIZE;
            fetch = fetchFunc;
            
            cudaMallocHost(&registry, sizeof(DataPoint) * EPOCH_SIZE);
            for (uint32 i = 0; i < EPOCH_SIZE; i++) {
                registry[i].offsetID = i;
            }
            
            remainedData = EPOCH_SIZE;
        }
        
        void fetchTrain(Data** dataset, uint32 step);
        void fetchTest(Tensor* data, Tensor* label, uint32 testID);
    };
    
} // seio

#endif //CRUSADER_DATAFETCHER_CUH
