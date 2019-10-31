//
// Created by Gael Aglin on 2019-10-06.
//

#ifndef DL85_DL85_H
#define DL85_DL85_H

#include <string>
using namespace std;

string search ( int argc, char *argv[], int* supports, int ntransactions, int nattributes, int nclasses, int *data, int *target, float maxError, bool stopAfterError, bool iterative );

#endif //DL85_DL85_H
