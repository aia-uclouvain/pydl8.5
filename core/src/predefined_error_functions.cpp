#include "predefined_error_functions.h"

float sse_tids_error(RCover* cover) {
    RCover::iterator it;

    // Computing the mean of the targets
    double sum = 0.;
    int count = 0;
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        sum += cover->dm->getY()[idx];
        count += 1;
    }

    double centroid = sum/count;
    float sse = 0.;

    // summing up squared errors to the centroid
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        float delta = cover->dm->getY()[idx] - centroid;
        sse += delta*delta;
    }    

    return sse;
}