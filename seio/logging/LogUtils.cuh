//
// Created by DanielSun on 3/14/2022.
//

#ifndef CRUSADER_LOGUTILS_CUH
#define CRUSADER_LOGUTILS_CUH

#include <string>
#include "Color.cuh"

using namespace std;

namespace seio {
    enum LogLevel {
        LOG_LEVEL_DEBUG = 0,
        LOG_LEVEL_INFO = 1,
        LOG_LEVEL_WARN = 2,
        LOG_LEVEL_ERROR = 3,
        LOG_LEVEL_FATAL = 4
    };

    enum LogSegments {
        LOG_SEG_SEIO = 0,
        LOG_SEG_SEANN = 1,
        LOG_SEG_SEBLAS = 2,
    };

    enum LogColor {
        LOG_COLOR_RED,
        LOG_COLOR_LIGHT_RED,
        LOG_COLOR_GREEN,
        LOG_COLOR_LIGHT_GREEN,
        LOG_COLOR_YELLOW,
        LOG_COLOR_LIGHT_YELLOW,
        LOG_COLOR_BLUE,
        LOG_COLOR_LIGHT_BLUE,
        LOG_COLOR_AQUA,
        LOG_COLOR_LIGHT_AQUA,
        LOG_COLOR_PURPLE,
        LOG_COLOR_LIGHT_PURPLE,
        LOG_COLOR_WHITE,
        LOG_COLOR_BRIGHT_WHITE,
    };

    void printLogHead(LogLevel level, LogSegments segment);

    void logInfo(LogSegments seg, string msg);

    void logInfo(LogSegments seg, const string &msg, LogColor color);

    void logDebug(LogSegments seg, string msg);

    void logDebug(LogSegments seg, const string &msg, LogColor color);

    void logWarn(LogSegments seg, string msg);

    void logError(LogSegments seg, string msg);

    void logFatal(LogSegments seg, string msg);

    void logProc(unsigned int proc, unsigned int finish);

    void logTrainingProcess(unsigned int batchId, unsigned int epochId, unsigned int batches
            , unsigned int epochs, float loss, float acc, float epochLoss, float epochAcc);
}


#endif //CRUSADER_LOGUTILS_CUH
