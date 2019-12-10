//
// Created by Gael Aglin on 2019-10-06.
//

#ifndef DL85_DL85_H
#define DL85_DL85_H

#include <string>
#include <map>
#include <vector>
#include <utility>
#include "globals.h"
using namespace std;

//string search ( int argc, char *argv[], int* supports, int ntransactions, int nattributes, int nclasses, int *data, int *target, float maxError, bool stopAfterError, bool iterative );
string search (//std::function<float(int*)> callback,
        //std::function<float(Array<int>::iterator)> callback,
        std::function<vector<float>(Array<int>*)> callback,
        int* supports,
        int ntransactions,
        int nattributes,
        int nclasses,
        int *data,
        int *target,
        float maxError,
        bool stopAfterError,
        bool iterative,
        bool user,
        int maxdepth = 1,
        int minsup = 1,
        bool infoGain = false,
        bool infoAsc = true,
        bool repeatSort = false,
        int timeLimit = 0,
        map<int, pair<int, int>>* continuousMap = NULL,
        bool save = false,
        bool nps_param = false,
        bool verbose_param = false);

#endif //DL85_DL85_H
