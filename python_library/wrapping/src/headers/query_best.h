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
  //Array<pair<bool,Attribute> > children;
  int nTransactions; //we do not want to count at new the support of an already viewed itemset
};


class Query_Best : public Query {
public:
    Query_Best ( Trie *trie, Data *data, ExpError *experror, int timeLimit, bool continuous, function<vector<float>(Array<int>*)>* error_callback = nullptr, function<vector<float>(Array<int>*)>* fast_error_callback = nullptr, bool predictor = false, float maxError = NO_ERR, bool stopAfterError = false );

    virtual ~Query_Best ();
    bool canimprove ( QueryData *left, Error ub );
    bool canSkip ( QueryData *actualBest);
    string printResult ( Data *data );
    virtual void printTimeOut(string*);
    string printResult ( Data *data2, QueryData_Best *data );
    virtual void printAccuracy ( Data *data2, QueryData_Best *data, string* );
    virtual Class runResult ( Data *data, Transaction transaction );
    virtual Class runResult ( QueryData_Best *node, Data *data, Transaction transaction );
    QueryData_Best *rootBest () const { return (QueryData_Best*) realroot->data; }
protected:
    int printResult ( QueryData_Best *node, int depth, string* );
    ExpError *experror;
};

#endif
