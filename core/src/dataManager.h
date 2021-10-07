//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RSBS_DATAMANAGER_H
#define RSBS_DATAMANAGER_H

#include <bitset>
#include "globals.h"

using namespace std;

#define M 64


class DataManager {

public:
    int nWords;

    DataManager(Supports supports, Transaction ntransactions, Attribute nattributes, Class nclasses, Bool *data, Class *target);

    ~DataManager(){
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

    bitset<M> * getAttributeCover(Attribute attr);

    bitset<M> * getClassCover(Class clas);

    /// get number of transactions
    [[nodiscard]] Transaction getNTransactions () const { return ntransactions; }

    /// get number of features
    [[nodiscard]] Attribute getNAttributes () const { return nattributes; }

    /// get number of transactions
    [[nodiscard]] Class getNClasses () const { return nclasses; }

    /// get array of support of each class
    [[nodiscard]] Supports getSupports () const { return supports; }

private:
    bitset<M> **b; /// matrix of data
    bitset<M> **c; /// vector of target
    Transaction ntransactions; /// number of transactions
    Attribute nattributes; /// number of features
    Class nclasses; /// number of classes
    Supports supports; /// array of support for each class

};


#endif //RSBS_DATAMANAGER_H
