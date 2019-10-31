#ifndef QUERY_BEST_H
#define QUERY_BEST_H
#include "trie.h"
#include "experror.h"
#include <query.h>

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
    Query_Best ( Trie *trie, Data *data, ExpError *experror, int timeLimit, bool continuous, float maxError = NO_ERR, bool stopAfterError = false );

    virtual ~Query_Best ();
    bool canimprove ( QueryData *left, Error ub );
    bool canSkip ( QueryData *actualBest);
    string printResult ( Data *data );
    virtual void printTimeOut(string*);
    string printResult ( Data *data2, QueryData_Best *data );
    virtual void printAccuracy ( Data *data2, QueryData_Best *data, string* );
    virtual Class runResult ( Data *data, Transaction transaction );
    virtual Class runResult ( QueryData_Best *node, Data *data, Transaction transaction );
    //bool updateMyData ( QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right);
    //QueryData  *initMyData ( pair<Supports,Support> supports, Error initBound, Support minsup);
    QueryData_Best *rootBest () const { return (QueryData_Best*) realroot->data; }
protected:
    int printResult ( QueryData_Best *node, int depth, string* );
    ExpError *experror;
};

#endif
