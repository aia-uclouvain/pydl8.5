#ifndef QUERY_BEST_H
#define QUERY_BEST_H

#include "trie.h"
#include <query.h>
#include <vector>

struct QueryData_Best {
    Attribute test;
    QueryData_Best *left, *right;
    Error leafError;
    Error error;
    Error lowerBound;
    Size size;

    QueryData_Best() {
        test = -1;
        left = nullptr;
        right = nullptr;
        leafError = FLT_MAX;
        error = FLT_MAX;
        lowerBound = 0;
        size = 1;
    }
};


class Query_Best : public Query {
public:
    Query_Best(Support minsup,
               Depth maxdepth,
               Trie *trie,
               DataManager *data,
               int timeLimit,
               function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
               function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
               function<float(RCover *)> *tids_error_callback = nullptr,
               float maxError = NO_ERR,
               bool stopAfterError = false);

    virtual ~Query_Best();

    inline bool canimprove(QueryData *left, Error ub) {
        return ((QueryData_Best *) left)->error < ub;
    }

    inline bool canSkip(QueryData *actualBest) {
        return floatEqual(((QueryData_Best *) actualBest)->error, ((QueryData_Best *) actualBest)->lowerBound);
    }

    void printResult(Tree *tree);

//    virtual void printTimeOut(Tree* tree );
    void printResult(QueryData_Best *data, Tree *tree);

    inline QueryData_Best *rootBest() const { return (QueryData_Best *) realroot->data; }

    virtual Error getTrainingError(const string &tree_json) {}

protected:
    int printResult(QueryData_Best *node_data, int depth, Tree *tree);

};

#endif
