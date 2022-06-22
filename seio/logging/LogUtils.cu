//
// Created by DanielSun on 3/14/2022.
//

#include <utility>
#include "LogUtils.cuh"

namespace seio {
    void printLogHead(LogLevel level, LogSegments segment){

        dye::colorful<basic_string<char>> segName;
        switch(segment){
            case LogSegments::LOG_SEG_SEANN:
                segName = dye::bright_white("seann");
                break;
            case LogSegments::LOG_SEG_SEIO:
                segName = dye::white("seio");
                break;
            case LogSegments::LOG_SEG_SEBLAS:
                segName = dye::light_yellow("seblas");
                break;
        }

        dye::colorful<basic_string<char>> levelPrefix;
        switch(level){
            case LogLevel::LOG_LEVEL_DEBUG:
                levelPrefix = dye::light_purple("DEBUG");
                break;
            case LogLevel::LOG_LEVEL_INFO:
                levelPrefix = dye::light_blue("INFO");
                break;
            case LogLevel::LOG_LEVEL_WARN:
                levelPrefix = dye::light_yellow("WARN");
                break;
            case LogLevel::LOG_LEVEL_ERROR:
                levelPrefix = dye::red("ERROR");
                break;
            case LogLevel::LOG_LEVEL_FATAL:
                levelPrefix = dye::red("FATAL");
                break;
        }

        time_t secs = time(nullptr);
        struct tm *local = localtime(&secs);

        //print the current time
        cout<<dye::light_blue("[")<<dye::blue(local->tm_hour)<<dye::bright_white(":")
        <<dye::blue(local->tm_min)<<dye::bright_white(":")<<dye::blue(local->tm_sec);

        //print the log segment
        cout<<dye::light_blue("|")<<segName<<dye::light_blue("]");

        //print the log level
        if(level == LogLevel::LOG_LEVEL_ERROR || level == LogLevel::LOG_LEVEL_FATAL)
            cout<<dye::red(": ")<<levelPrefix<<dye::red(" >>> ");
        else
            cout<<dye::light_purple(": ")<<levelPrefix<<dye::bright_white(" >>> ");
    }

    void printColored(const string& msg, LogColor color){
        switch(color){
            case LOG_COLOR_RED:
                cout<<dye::red(msg);
                break;
            case LOG_COLOR_GREEN:
                cout<<dye::green(msg);
                break;
            case LOG_COLOR_YELLOW:
                cout<<dye::yellow(msg);
                break;
            case LOG_COLOR_BLUE:
                cout<<dye::blue(msg);
                break;
            case LOG_COLOR_PURPLE:
                cout<<dye::purple(msg);
                break;
            case LOG_COLOR_AQUA:
                cout<<dye::aqua(msg);
                break;
            case LOG_COLOR_WHITE:
                cout<<dye::white(msg);
                break;
            case LOG_COLOR_LIGHT_RED:
                cout<<dye::light_red(msg);
                break;
            case LOG_COLOR_LIGHT_GREEN:
                cout<<dye::light_green(msg);
                break;
            case LOG_COLOR_LIGHT_YELLOW:
                cout<<dye::light_yellow(msg);
                break;
            case LOG_COLOR_LIGHT_BLUE:
                cout<<dye::light_blue(msg);
                break;
            case LOG_COLOR_LIGHT_PURPLE:
                cout<<dye::light_purple(msg);
                break;
            case LOG_COLOR_LIGHT_AQUA:
                cout<<dye::light_aqua(msg);
                break;
            case LOG_COLOR_BRIGHT_WHITE:
                cout<<dye::bright_white(msg);
                break;
            default:
                cout<<msg;
                break;
        }
    }

    void logInfo(LogSegments segment, string msg){
        printLogHead(LogLevel::LOG_LEVEL_INFO, segment);
        cout<<dye::blue(std::move(msg))<<endl;
    }

    void logInfo(LogSegments seg, const string& msg, LogColor color){
        printLogHead(LogLevel::LOG_LEVEL_INFO, seg);
        printColored(msg, color);
        cout<<endl;
    }

    void logDebug(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_DEBUG, seg);
        cout<<dye::purple(std::move(msg))<<endl;
    }

    void logDebug(LogSegments seg, const string& msg, LogColor color){
        printLogHead(LogLevel::LOG_LEVEL_DEBUG, seg);
        printColored(msg, color);
        cout<<endl;
    }

    void logWarn(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_WARN, seg);
        cout<<dye::yellow(std::move(msg))<<endl;
    }

    void logError(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_ERROR, seg);
        cout<<dye::red(std::move(msg))<<endl;
    }

    void logFatal(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_FATAL, seg);
        cout<<dye::red(std::move(msg))<<endl;
    }
}