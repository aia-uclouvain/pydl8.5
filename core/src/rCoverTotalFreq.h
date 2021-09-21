//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RSBS_RCOVER_TOTAL_FREQ_H
#define RSBS_RCOVER_TOTAL_FREQ_H

#include "rCover.h"

class RCoverTotalFreq : public RCover {

public:

    RCoverTotalFreq(DataManager* dmm);

    ~RCoverTotalFreq(){}

    void intersect(Attribute attribute, bool positive = true);

    pair<Supports, Support> temporaryIntersect(Attribute attribute, bool positive = true);

    Supports getSupportPerClass();

    Supports getSupportPerClass(bitset<M>** cover, int nValidWords, int* validIndexes);

    SupportClass countSupportClass(bitset<M>& coverWord, int wordIndex);

};


#endif //RSBS_RCOVER_TOTAL_FREQ_H
