#ifndef QUERY_TOTALFREQ_H
#define QUERY_TOTALFREQ_H
#include <query_best.h>
#include <vector>


class Query_TotalFreq : public Query_Best {
public:
    Query_TotalFreq( Trie *trie, Data *data, ExpError *experror, int timeLimit, bool continuous, function<float(Array<int>*)>* error_callback = nullptr, function<vector<float>(Array<int>*)>* fast_error_callback = nullptr, float maxError = NO_ERR, bool stopAfterError = false );

    ~Query_TotalFreq();
    bool is_freq ( pair<Supports,Support> supports ); 
    bool is_pure ( pair<Supports,Support> supports );
    bool updateData ( QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right);
    QueryData *initData ( Array<Transaction> tid, Data *dataReader, Error initBound, Support minsup, Depth currentMaxDepth = -1);
    //QueryData *initDataFromUser ( Array<Transaction> tid, Error initBound, Support minsup, Depth currentMaxDepth = -1);
    //QueryData *initDataFastFromUser ( pair<Supports,Support> supports, Error parent_ub, Support minsup, Depth currentMaxDepth = -1);
    void printAccuracy ( Data *data2, QueryData_Best *data, string* );
protected:
    int printResult ( TrieNode *node, int depth, string* );
};

#endif
