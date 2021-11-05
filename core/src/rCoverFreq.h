//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RSBS_RCOVER_TOTAL_FREQ_H
#define RSBS_RCOVER_TOTAL_FREQ_H

#include "rCover.h"

class RCoverFreq : public RCover {

public:

    RCoverFreq(DataManager* dmm);

    ~RCoverFreq(){}

    void intersect(Attribute attribute, bool positive = true);

    pair<Supports, Support> temporaryIntersect(Attribute attribute, bool positive = true);

    Supports getSupportPerClass();

    Supports getSupportPerClass(bitset<M>** cover, int nValidWords, int* validIndexes);

    SupportClass countSupportClass(bitset<M>& coverWord, int wordIndex);

};

/*class RCoverFreq : public RCover {

public:

    RCoverFreq(DataManager* dmm);

    ~RCoverFreq(){}

    void intersect(Attribute attribute, bool positive = true);

    pair<Supports, Support> temporaryIntersect(Attribute attribute, bool positive = true);

    Supports getSupportPerClass();

    Supports getSupportPerClass(ulong * cover, int nValidWords, int* validIndexes);

    SupportClass countSupportClass(ulong coverWord, int wordIndex);

};*/

#endif //RSBS_RCOVER_TOTAL_FREQ_H
