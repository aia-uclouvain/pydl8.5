//
// Created by Gael Aglin on 2019-10-08.
//

#include "dataBinaryPython.h"
#include <iostream>

using namespace std;

DataBinaryPython::DataBinaryPython(Supports supportss, Transaction numtransactions, Attribute numattributes, Class numclasses, Bool *data, Class *target) {
    b = (int **)malloc(numtransactions * sizeof(int *));
    for (int i = 0; i < numtransactions; i++){
        //b[i] = (int *)malloc(numattributes * sizeof(int));
        b[i] = &data[i * numattributes];
    }
    c = target;
    nclasses = numclasses;
    ntransactions = numtransactions;
    nattributes = numattributes;
    supports = supportss;

    ::nattributes = nattributes;
    ::nclasses = nclasses;
}


DataBinaryPython::~DataBinaryPython() {
    for (int i = 0; i < ntransactions; ++i) {
        delete[] b[i];
    }
    //delete[] b[0];
    delete[] b;
    delete[] c;
}

void DataBinaryPython::read ( const char *filename = "" ) {

}
