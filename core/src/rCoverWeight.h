//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RCOVER_WEIGHT_H
#define RCOVER_WEIGHT_H

#include "rCover.h"

class RCoverWeight : public RCover {

public:

    RCoverWeight(DataManager* dmm, vector<float>* weights);

    RCoverWeight(RCoverWeight &&cover, vector<float>* weights);

    ~RCoverWeight(){}

    void intersect(Attribute attribute, bool positive = true);

    pair<ErrorVals, Support> temporaryIntersect(Attribute attribute, bool positive = true);

    ErrorVals getErrorValPerClass();

    ErrorVals getErrorValPerClass(bitset<M>* cover, int nValidWords, int* validIndexes);

    ErrorVal getErrorVal(bitset<M>& coverWord, int wordIndex);

    vector<int> getTransactionsID(bitset<M>& word, int real_word_index);

    pair<ErrorVal, Support> getSups(bitset<M>& word, int real_word_index);

    vector<int> getTransactionsID();

    vector<float>* weights;

};


#endif //RCOVER_WEIGHT_H
