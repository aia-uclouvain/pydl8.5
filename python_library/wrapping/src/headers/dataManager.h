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

    DataManager(int* supports, int ntransactions, int nattributes, int nclasses, int *b, int *c);

    ~DataManager(){
        delete[]b;
        delete[]c;
    }

    bitset<M> * getAttributeCover(int attr);

    bitset<M> * getClassCover(int clas);

    /// get number of transactions
    virtual int getNTransactions () const { return ntransactions; }

    /// get number of features
    virtual int getNAttributes () const { return nattributes; }

    /// get number of transactions
    int getNClasses () const { return nclasses; }

    /// get array of support of each class
    Supports getSupports () const { return supports; }

private:
    bitset<M> **b; /// matrix of data
    bitset<M> **c; /// vector of target
    Transaction ntransactions; /// number of transactions
    Attribute nattributes; /// number of features
    Class nclasses; /// number of classes
    Supports supports; /// array of support for each class

};


#endif //RSBS_DATAMANAGER_H
