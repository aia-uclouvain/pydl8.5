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

    void intersect(Attribute attribute, const vector<float>* weights, bool positive = true);

    pair<Supports, Support> temporaryIntersect(Attribute attribute, const vector<float>* weights, bool positive = true);

    Supports getSupportPerClass(const vector<float>* weights);

};


#endif //RSBS_RCOVER_TOTAL_FREQ_H
