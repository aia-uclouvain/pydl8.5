

#ifndef DL8_NEW_LOGGER_H
#define DL8_NEW_LOGGER_H

#include <iostream>
#include <string>
#include "globals.h"

using namespace std;

class Logger {

public:
    //static const bool enable = false;
    static void writeConsole(string s);
    static void writeConsoleAndReturn(string s);

    template<typename T>
    static void showMessageAndReturn(T &&t) {
        if (verbose){
            std::cout << t << "\n";
        }
    }

    template<typename Head, typename... Tail>
    static void showMessageAndReturn(Head &&head, Tail&&... tail) {
        if (verbose){
            std::cout << head;
            showMessageAndReturn(std::forward<Tail>(tail)...);
        }
    }

    template<typename T>
    static void showMessage(T &&t) {
        if (verbose){
            std::cout << t;
        }
    }

    template<typename Head, typename... Tail>
    static void showMessage(Head &&head, Tail&&... tail) {
        if (verbose){
            std::cout << head;
            showMessage(std::forward<Tail>(tail)...);
        }
    }

};


#endif //DL8_NEW_LOGGER_H
