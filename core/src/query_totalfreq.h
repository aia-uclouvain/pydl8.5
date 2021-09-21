#ifndef QUERY_TOTALFREQ_H
#define QUERY_TOTALFREQ_H

#include <query_best.h>
#include <vector>

class Query_TotalFreq : public Query_Best {
public:
    Query_TotalFreq(Support minsup,
                    Depth maxdepth,
                    Trie *trie,
                    DataManager *data,
                    int timeLimit,
                    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
                    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
                    function<float(RCover *)> *tids_error_callback = nullptr,
                    float maxError = NO_ERR,
                    bool stopAfterError = false);

    ~Query_TotalFreq();

    bool is_freq(pair<Supports, Support> supports);

    bool is_pure(pair<Supports, Support> supports);

    bool updateData(QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right);

    QueryData *initData(RCover *tid, Depth currentMaxDepth = -1);

    LeafInfo computeLeafInfo(RCover *cover);

    LeafInfo computeLeafInfo(Supports itemsetSupport);

protected:
};

#endif