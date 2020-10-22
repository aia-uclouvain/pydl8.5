//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RSBS_RCOVER_WEIGHTED_H
#define RSBS_RCOVER_WEIGHTED_H

#include "rCover.h"

class RCoverWeighted : public RCover {

public:

    RCoverWeighted(DataManager* dmm);

    RCoverWeighted(RCoverWeighted &&cover)  noexcept ;

    ~RCoverWeighted(){}

    void intersect(Attribute attribute, const vector<float>* weights, bool positive = true);

    pair<Supports, Support> temporaryIntersect(Attribute attribute, const vector<float>* weights, bool positive = true);

    Supports getSupportPerClass(const vector<float>* weights);

    vector<int> getTransactionsID(bitset<M>& word, int real_word_index);

    pair<SupportClass, Support> getSups(bitset<M>& word, int real_word_index, const vector<float>* weights);

    vector<int> getTransactionsID();

};


#endif //RSBS_RCOVER_WEIGHTED_H
