#ifndef QUERY_BEST_H
#define QUERY_BEST_H

#include "trie.h"
#include <query.h>
#include <vector>

struct QueryData_Best {
    Attribute* tests;
    QueryData_Best** lefts;
    QueryData_Best** rights;
    Error* leafErrors;
    Error* errors;
    Error* lowerBounds;
    Size* sizes;
    int n_quantiles;

    QueryData_Best(int n_quantiles): n_quantiles(n_quantiles) {
        tests = new Attribute[n_quantiles];
        lefts = new QueryData_Best*[n_quantiles];
        rights = new QueryData_Best*[n_quantiles];
        // leafError = FLT_MAX;
        // error = FLT_MAX;
        // lowerBound = 0;
        leafErrors = new Error[n_quantiles];
        errors = new Error[n_quantiles];
        lowerBounds = new Error[n_quantiles];
        sizes = new Size[n_quantiles];

        for (int i = 0; i < n_quantiles; i++) {
            tests[i] = -1;
            lefts[i] = nullptr;
            rights[i] = nullptr;
            leafErrors[i] = FLT_MAX;
            errors[i] = FLT_MAX;
            lowerBounds[i] = 0;
            sizes[i] = 1;
        }
    }

    virtual ~QueryData_Best(){
        delete[] tests;
        delete[] leafErrors;
        delete[] errors;
        delete[] lowerBounds;
        delete[] lefts; 
        delete[] rights;
        delete[] sizes;
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
               float* maxError = nullptr,
               bool* stopAfterError = nullptr);

    virtual ~Query_Best();

    bool canimprove(QueryData *left, Error* ub, int n_quantiles) {
        Error* errors = ((QueryData_Best *) left)->errors;
        for (int i = 0; i < n_quantiles; i++) {
            if (errors[i] < ub[i]) 
                return true;
        }
        return false;
        // return ((QueryData_Best *) left)->error < ub;
    }

    bool canSkip(QueryData *actualBest, int n_quantiles) {
        Error* errors = ((QueryData_Best *) actualBest)->errors;
        Error* lowerBounds = ((QueryData_Best *) actualBest)->lowerBounds;
        for (int i = 0; i < n_quantiles; i++) {
            if (!floatEqual(errors[i], lowerBounds[i]))
                return false;
        }
        return true;
    }

    void printResult(Tree *tree, int quantile_idx = 0);

//    virtual void printTimeOut(Tree* tree );
    void printResult(QueryData_Best *data, Tree *tree, int quantile_idx = 0);

    inline QueryData_Best *rootBest() const { return (QueryData_Best *) realroot->data; }

    virtual Error getTrainingError(const string &tree_json) {}

protected:
    int printResult(QueryData_Best *node_data, int depth, Tree *tree, int quantile_idx = 0);

};

#endif
