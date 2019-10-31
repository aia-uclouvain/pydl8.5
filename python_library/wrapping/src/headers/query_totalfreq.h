#ifndef QUERY_TOTALFREQ_H
#define QUERY_TOTALFREQ_H
#include <query_best.h>


class Query_TotalFreq : public Query_Best {
public:
    Query_TotalFreq( Trie *trie, Data *data, ExpError *experror, int timeLimit, bool continuous, float maxError = NO_ERR, bool stopAfterError = false );

    ~Query_TotalFreq();
    bool is_freq ( pair<Supports,Support> supports ); 
    bool is_pure ( pair<Supports,Support> supports );
    bool updateData ( QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right);
    QueryData *initData ( pair<Supports,Support> supports, Error initBound, Support minsup, Depth currentMaxDepth = -1);
    void printAccuracy ( Data *data2, QueryData_Best *data, string* );
protected:
    int printResult ( TrieNode *node, int depth, string* );
};

#endif
