//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RCOVER_FREQ_H
#define RCOVER_FREQ_H

#include "rCover.h"

class RCoverFreq : public RCover {

public:

    RCoverFreq(DataManager* dmm);

    ~RCoverFreq(){}

    void intersect(Attribute attribute, bool positive = true);

    pair<ErrorVals, Support> temporaryIntersect(Attribute attribute, bool positive = true);

    ErrorVals getErrorValPerClass();

    ErrorVals getErrorValPerClass(bitset<M>* cover, int nValidWords, int* validIndexes);

    ErrorVal getErrorVal(bitset<M>& coverWord, int wordIndex);

    ErrorVal diffErrorVal(bitset<M>* cover1, bitset<M>* cover2);

};

#endif //RCOVER_FREQ_H
