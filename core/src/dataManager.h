//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <bitset>
#include "globals.h"

using namespace std;

#define M 64

class DataManager {

public:
    int nWords;

    DataManager(ErrorVals supports, Transaction _ntransactions, Attribute _nattributes, Class _nclasses, Bool *data, Class *target);

    ~DataManager(){
        for (int i = 0; i < nattributes_; ++i) {
            delete[] b[i];
        }
        delete[]b;
        for (int j = 0; j < nclasses_; ++j) {
            delete[] c[j];
        }
        delete[]c;
    }

    bitset<M> * getAttributeCover(Attribute attr);

    bitset<M> * getClassCover(Class clas);

    /// get number of transactions
    [[nodiscard]] Transaction getNTransactions () const { return ntransactions_; }

    /// get number of features
    [[nodiscard]] Attribute getNAttributes () const { return nattributes_; }

    /// get number of transactions
    [[nodiscard]] Class getNClasses () const { return nclasses_; }

    /// get array of support of each class
    [[nodiscard]] ErrorVals getSupports () const { return supports; }

private:
    bitset<M> **b; /// matrix of data
    bitset<M> **c; /// vector of target
    Transaction ntransactions_; /// number of transactions
    Attribute nattributes_; /// number of features
    Class nclasses_; /// number of classes
    ErrorVals supports; /// array of support for each class

};

#endif //DATAMANAGER_H
