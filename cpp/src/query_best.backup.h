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
  Error lowerBound;
  Size size;
  Depth solutionDepth;
  Supports corrects, falses;
  //Array<Attribute> successors;
    ~QueryData_Best(){
        if (corrects) deleteSupports(corrects);
        if (falses) deleteSupports(falses);
        //if (successors.elts) successors.free();
    }
};


class Query_Best : public Query {
public:
    Query_Best ( Trie *trie,
            DataManager *data,
            ExpError *experror,
            int timeLimit,
            bool continuous,
            function<vector<float>(RCover*)>* tids_error_class_callback = nullptr,
            function<vector<float>(RCover*)>* supports_error_class_callback = nullptr,
            function<float(RCover*)>*  tids_error_callback = nullptr,
            float maxError = NO_ERR,
            bool stopAfterError = false );

    virtual ~Query_Best ();
    bool canimprove ( QueryData *left, Error ub );
    bool canSkip ( QueryData *actualBest);
    void printResult ( DataManager *data, Tree* tree );
    virtual void printTimeOut(Tree* tree );
    void printResult ( DataManager *data2, QueryData_Best *data, Tree* tree );
    virtual void printAccuracy ( DataManager *data2, QueryData_Best *data, Tree* );
    //virtual Class runResult ( DataManager *data, Transaction transaction );
    //virtual Class runResult ( QueryData_Best *node, DataManager *data, Transaction transaction );
    QueryData_Best *rootBest () const { return (QueryData_Best*) realroot->data; }
protected:
    int printResult ( QueryData_Best *node, int depth, Tree* tree );
    ExpError *experror;
};

#endif
