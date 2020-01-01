#ifndef QUERY_TOTALFREQ_H
#define QUERY_TOTALFREQ_H
#include <query_best.h>
#include <vector>


class Query_TotalFreq : public Query_Best {
public:
    Query_TotalFreq( Trie *trie, DataManager *data, ExpError *experror, int timeLimit, bool continuous, function<vector<float>(RCover*)>* error_callback = nullptr, function<vector<float>(RCover*)>* fast_error_callback = nullptr, function<float(RCover*)>*  predictor_error_callback = nullptr, float maxError = NO_ERR, bool stopAfterError = false );

    ~Query_TotalFreq();
    bool is_freq ( pair<Supports,Support> supports );
    bool is_pure ( pair<Supports,Support> supports );
    bool updateData ( QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right);
    QueryData *initData ( RCover* tid, Error initBound, Support minsup, Depth currentMaxDepth = -1);
    //QueryData *initData ( Array<Transaction> tid, Error initBound, Support minsup, Depth currentMaxDepth = -1);
    void printAccuracy ( DataManager *data2, QueryData_Best *data, string* );
protected:
    int printResult ( TrieNode *node, int depth, string* );
};

#endif