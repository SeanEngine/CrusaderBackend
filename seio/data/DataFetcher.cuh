//
// Created by Dylan on 7/24/2022.
//

#ifndef CRUSADER_DATAFETCHER_CUH
#define CRUSADER_DATAFETCHER_CUH

#include <utility>
#include <filesystem>

#include "Data.cuh"
#include "DataLoader.cuh"

namespace seio {
    
    struct DataPoint {
        uint32 offsetID;
        char* elementName;
        uint32 nameSize;
        bool WITH_LOCATION = false;
    };
    
    class DataFetcher {
    public:
        string ROOT_PATH;
        string DATASET_NAME;
        string DATA_SUBFD;
        string LABEL_SUBFD;
        uint32 EPOCH_SIZE;
        uint32 TEST_SIZE;
        Loader* dataLoader;
        uint32 remainedData;
        DataPoint* registry{};
        
        //for datasets of single file
        DataFetcher(string rootPath, string datasetName, uint32 EPOCH_SIZE, uint32 TEST_SIZE, Loader* dataLoader) {
            ROOT_PATH = std::move(rootPath);
            DATASET_NAME = std::move(datasetName);
            this->EPOCH_SIZE = EPOCH_SIZE;
            this->TEST_SIZE = TEST_SIZE;
            this->dataLoader = dataLoader;
            
            cudaMallocHost(&registry, sizeof(DataPoint) * (EPOCH_SIZE + TEST_SIZE));
            for (uint32 i = 0; i < EPOCH_SIZE + TEST_SIZE; i++) {
                registry[i].offsetID = i;
            }
            
            remainedData = EPOCH_SIZE + TEST_SIZE;
        }
        
        //for datasets of multiple files
        // DATA_SUBFDR : The subfolder for data
        // LABEL_SUBFDR : The subfolder for label
        DataFetcher(string rootPath, string DATA_SUBFDR, string LABEL_SUBFDR,
                    uint32 EPOCH_SIZE, uint32 TEST_SIZE, Loader* dataLoader) :
                    ROOT_PATH(std::move(rootPath)), DATA_SUBFD(std::move(DATA_SUBFDR)),
                    LABEL_SUBFD(std::move(LABEL_SUBFDR)), EPOCH_SIZE(EPOCH_SIZE),
                    TEST_SIZE(TEST_SIZE), dataLoader(dataLoader) {
            
            uint32 runningID = 0;
            cudaMallocHost(&registry, sizeof(DataPoint) * (EPOCH_SIZE + TEST_SIZE));
            for (auto const& dir_entry : std::filesystem::directory_iterator{ROOT_PATH +
            "\\" + DATA_SUBFD}) {
                if(runningID < EPOCH_SIZE + TEST_SIZE) {
                    uint32 strSize = dir_entry.path().filename().string().length();
                    cudaMallocHost(&(registry[runningID].elementName), sizeof(char) *
                                                                       strSize);
                    cudaMemcpy(registry[runningID].elementName, dir_entry.path().filename().string().c_str(),
                               sizeof(char) * strSize, cudaMemcpyHostToHost);
                    assertCuda(__FILE__, __LINE__);
        
                    registry[runningID].offsetID = runningID;
                    registry[runningID].nameSize = strSize;
                    runningID++;
                }
            }
            logInfo(LOG_SEG_SEIO,"DataFetcher Found " +
                std::to_string(runningID) + " data files", LOG_COLOR_LIGHT_YELLOW);
            remainedData = EPOCH_SIZE + TEST_SIZE;
        }
    
        //for datasets of multiple files and without independent label files
        DataFetcher(string rootPath,uint32 EPOCH_SIZE, uint32 TEST_SIZE, Loader* dataLoader) :
                    ROOT_PATH(std::move(rootPath)), EPOCH_SIZE(EPOCH_SIZE),
                    TEST_SIZE(TEST_SIZE), dataLoader(dataLoader) {
            
            uint32 runningID = 0;
            cudaMallocHost(&registry, sizeof(DataPoint) * (EPOCH_SIZE + TEST_SIZE));
            for (auto const& dir_entry : std::filesystem::directory_iterator{ROOT_PATH}) {
                if(runningID < EPOCH_SIZE + TEST_SIZE) {
                    uint32 strSize = dir_entry.path().filename().string().length();
                    cudaMallocHost(&(registry[runningID].elementName), sizeof(char) *
                            (strSize) + 1);
                    cudaMemcpy(registry[runningID].elementName, dir_entry.path().filename().string().c_str(),
                               sizeof(char) * strSize, cudaMemcpyHostToHost);
                    assertCuda(__FILE__, __LINE__);
                    
                    registry[runningID].offsetID = runningID;
                    registry[runningID].nameSize = strSize;
                    registry[runningID].elementName[strSize] = '\0';
                    registry[runningID].WITH_LOCATION = true;
                    runningID++;
                }
            }
            logInfo(LOG_SEG_SEIO,"DataFetcher Found " +
                std::to_string(runningID) + " data files", LOG_COLOR_LIGHT_YELLOW);
            
            remainedData = EPOCH_SIZE + TEST_SIZE;
        }
    
        void fetchTrain(Data** dataset, uint32 step);
        void fetchTest(Tensor* data, Tensor* label);
    };
    
} // seio

#endif //CRUSADER_DATAFETCHER_CUH
