//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef DATAHANDLER_H
#define DATAMANAGER_H

#include "globals.h"

using namespace std;

#define M 64


class DataHandler {

public:
    int nWords;

    DataHandler(Supports supports, Transaction ntransactions, Attribute nattributes, Class nclasses, Bool *data, Class *target);

    ~DataHandler(){
        for (int i = 0; i < nattributes; ++i) {
            delete[] b[i];
        }
        delete[]b;
        for (int j = 0; j < nclasses; ++j) {
            delete[] c[j];
        }
        delete[]c;
        deleteSupports(supports);
    }

    ulong * getAttributeCover(Attribute attr);

    ulong * getClassCover(Class clas);

    /// get number of transactions
    [[nodiscard]] Transaction getNTransactions () const { return ntransactions; }

    /// get number of features
    [[nodiscard]] Attribute getNAttributes () const { return nattributes; }

    /// get number of transactions
    [[nodiscard]] Class getNClasses () const { return nclasses; }

    /// get array of support of each class
    [[nodiscard]] Supports getSupports () const { return supports; }

private:
    ulong **b; /// matrix of data
    ulong **c; /// vector of target
    Transaction ntransactions; /// number of transactions
    Attribute nattributes; /// number of features
    Class nclasses; /// number of classes
    Supports supports; /// array of support for each class

};


#endif //DATAMANAGER_H
