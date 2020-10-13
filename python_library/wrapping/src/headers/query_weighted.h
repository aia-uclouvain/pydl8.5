#ifndef QUERY_WEIGHTED_H
#define QUERY_WEIGHTED_H

#include <query_best.h>
#include <vector>

class Query_Weighted : public Query_Best {
public:
    Query_Weighted(Trie *trie,
                   DataManager *data,
                   ExpError *experror,
                   int timeLimit,
                   bool continuous,
                   function<vector<float>(RCover *)>* tids_error_class_callback = nullptr,
                   function<vector<float>(RCover *)>* supports_error_class_callback = nullptr,
                   function<float(RCover *)>* tids_error_callback = nullptr,
                   function<vector<float>(string)>* example_weight_callback = nullptr,
                   function<float(string)>* predict_error_callback = nullptr,
                   float maxError = NO_ERR,
                   bool stopAfterError = false);

    ~Query_Weighted();

    bool is_freq(pair<Supports, Support> supports);

    bool is_pure(pair<Supports, Support> supports);

    bool updateData(QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right);

    QueryData *initData(RCover *tid, Depth currentMaxDepth = -1);

    ErrorValues computeErrorValues(RCover *cover);

    ErrorValues computeErrorValues(Supports itemsetSupport, bool onlyerror = false);

    Error computeOnlyError(Supports itemsetSupport);

    Error getTrainingError(const string& tree_json);

    vector<float> weights;
    function<vector<float>(string)>* example_weight_callback = nullptr;
    function<float(string)>* predict_error_callback = nullptr;

protected:
};

#endif