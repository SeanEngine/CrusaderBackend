//
// Created by Dylan on 7/19/2022.
//

#ifndef CRUSADER_TRAINPROCDISPLAY_CUH
#define CRUSADER_TRAINPROCDISPLAY_CUH

#include "../../seblas/operations/cuOperations.cuh"

#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#define DISABLE_NEWLINE_AUTO_RETURN  0x0008

using namespace seblas;
namespace seio {
    
    class ProcDisplay {
    public:
        
        const char* LOSS_TYPE;
        
        float epochTrainLoss = 0;
        float epochValidLoss = 0;
        float batchID = 0;
        
        char* paramBuf;
        
        ProcDisplay(const char* lossType) {
            LOSS_TYPE = lossType;
            cudaMallocHost(&paramBuf, 20 * sizeof(char));
    
            // enabling VT100 style in current console
            DWORD l_mode;
            HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
            GetConsoleMode(hStdout,&l_mode);
            SetConsoleMode( hStdout, l_mode |
                                     ENABLE_VIRTUAL_TERMINAL_PROCESSING |
                                     DISABLE_NEWLINE_AUTO_RETURN );
        }
        
        //called at beginning of epoch
        void show(uint32 epochID);
        
        void update(float trainLoss, float validLoss);
        
        static char* floatToString(float val, char* buf){
            uint32 size = to_string(val).size();
            for(uint32 i = 0; i < 20; i++){
                if(i < size){
                    buf[i] = to_string(val)[i];
                } else {
                    buf[i] = ' ';
                }
            }
            return buf;
        }
        
        void reset();
    };
    
} // seio

#endif //CRUSADER_TRAINPROCDISPLAY_CUH
