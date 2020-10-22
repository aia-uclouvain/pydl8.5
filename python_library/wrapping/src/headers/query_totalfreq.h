#ifndef QUERY_TOTALFREQ_H
#define QUERY_TOTALFREQ_H

#include <query_best.h>
#include <vector>

class Query_TotalFreq : public Query_Best {
public:
    Query_TotalFreq(Trie *trie,
                    DataManager *data,
                    ExpError *experror,
                    int timeLimit,
                    bool continuous,
                    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
                    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
                    function<float(RCover *)> *tids_error_callback = nullptr,
                    function<vector<float>()> *example_weight_callback = nullptr,
                    function<vector<float>(string)> *predict_error_callback = nullptr,
                    vector<float> *weights = nullptr,
                    float maxError = NO_ERR,
                    bool stopAfterError = false);

    ~Query_TotalFreq();

    bool is_freq(pair<Supports, Support> supports);

    bool is_pure(pair<Supports, Support> supports);

    bool updateData(QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right);

    QueryData *initData(RCover *tid, Depth currentMaxDepth = -1);

    ErrorValues computeErrorValues(RCover *cover);

    ErrorValues computeErrorValues(Supports itemsetSupport);

//    Error getTrainingError(const string& tree_json);

protected:
};

#endif