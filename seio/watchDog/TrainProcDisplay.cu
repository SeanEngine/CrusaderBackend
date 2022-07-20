//
// Created by Dylan on 7/19/2022.
//

#include "TrainProcDisplay.cuh"

namespace seio {
    
    void remove_all_chars(char* str, char c) {
        char *pr = str, *pw = str;
        while (*pr) {
            *pw = *pr++;
            pw += (*pw != c);
        }
        *pw = '\0';
    }
    
    void ProcDisplay::show(uint32 epochID) {
        logInfo(LOG_SEG_SEANN, "Training Epoch: " + to_string(epochID)
                               + "  Loss Type: " + string(LOSS_TYPE));

        cout<<"* Train Batch Loss *"<<"   ";
        cout<<"* Valid batch Loss *"<<"   ";
        cout<<"* Train Epoch Loss *"<<"   ";
        cout<<"* Valid Epoch Loss *"<<"   ";
        cout<<"* Batch ID * "<<endl;
        
    }
    
    void ProcDisplay::update(float trainLoss, float valLoss) {
        batchID++;
        epochTrainLoss = (epochValidLoss * (batchID - 1) + trainLoss) / batchID;
        epochValidLoss = (epochValidLoss * (batchID - 1) + valLoss) / batchID;
    
        //this overwriting is not working in clion console but works in terminal
        cout<<"\x1B[2K\r";
        remove_all_chars(floatToString(trainLoss, paramBuf), '\n');
        cout << paramBuf << "   ";
        remove_all_chars(floatToString(valLoss, paramBuf), '\n');
        cout << paramBuf << "   ";
        remove_all_chars(floatToString(epochTrainLoss, paramBuf), '\n');
        cout << paramBuf << "   ";
        remove_all_chars(floatToString(epochValidLoss, paramBuf), '\n');
        cout << paramBuf << "   ";
        cout << batchID;
    }
    
    void ProcDisplay::reset() {
        epochTrainLoss = 0;
        epochValidLoss = 0;
        batchID = 0;
        cout<<"\n"<<endl;
    }
} // seio