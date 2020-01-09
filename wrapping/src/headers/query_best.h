#ifndef QUERY_BEST_H
#define QUERY_BEST_H
#include "trie.h"
#include "experror.h"
#include <query.h>
#include <vector>

struct QueryData_Best {
  Attribute test;
  QueryData_Best *left, *right;
  Error leafError;
  Error error;
  Error initUb = NO_ERR;
  Error lowerBound;
  Size size;
  Depth solutionDepth;
  //Array<pair<bool,Attribute> > successors = nullptr;
};


class Query_Best : public Query {
public:
    Query_Best ( Trie *trie, DataManager *data, ExpError *experror, int timeLimit, bool continuous, function<vector<float>(RCover*)>* error_callback = nullptr, function<vector<float>(RCover*)>* fast_error_callback = nullptr, function<float(RCover*)>*  predictor_error_callback = nullptr, float maxError = NO_ERR, bool stopAfterError = false );

    virtual ~Query_Best ();
    bool canimprove ( QueryData *left, Error ub );
    bool canSkip ( QueryData *actualBest);
    string printResult ( DataManager *data );
    virtual void printTimeOut(string*);
    string printResult ( DataManager *data2, QueryData_Best *data );
    virtual void printAccuracy ( DataManager *data2, QueryData_Best *data, string* );
    //virtual Class runResult ( DataManager *data, Transaction transaction );
    //virtual Class runResult ( QueryData_Best *node, DataManager *data, Transaction transaction );
    QueryData_Best *rootBest () const { return (QueryData_Best*) realroot->data; }
protected:
    int printResult ( QueryData_Best *node, int depth, string* );
    ExpError *experror;
};

#endif
