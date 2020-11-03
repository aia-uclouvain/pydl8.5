//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RSBS_RCOVER_WEIGHTED_H
#define RSBS_RCOVER_WEIGHTED_H

#include "rCover.h"

class RCoverWeighted : public RCover {

public:

    RCoverWeighted(DataManager* dmm, vector<float>* weights);

    RCoverWeighted(RCoverWeighted &&cover, vector<float>* weights);

    ~RCoverWeighted(){}

    void intersect(Attribute attribute, bool positive = true);

    pair<Supports, Support> temporaryIntersect(Attribute attribute, bool positive = true);

    Supports getSupportPerClass();

    Supports getSupportPerClass(bitset<M>** cover, int nValidWords, int* validIndexes);

    SupportClass countSupportClass(bitset<M>& coverWord, int wordIndex);

    vector<int> getTransactionsID(bitset<M>& word, int real_word_index);

    pair<SupportClass, Support> getSups(bitset<M>& word, int real_word_index);

    vector<int> getTransactionsID();

    vector<float>* weights;

};


#endif //RSBS_RCOVER_WEIGHTED_H
