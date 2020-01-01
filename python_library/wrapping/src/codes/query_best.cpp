#include "query_best.h"
#include <iostream>
#include <dataContinuous.h>

using namespace std;

Query_Best::Query_Best(Trie *Trie, DataManager *data, ExpError *experror, int timeLimit, bool continuous, function<vector<float>(RCover*)>* error_callback, function<vector<float>(RCover*)>* fast_error_callback, function<float(RCover*)>*  predictor_error_callback, float maxError, bool stopAfterError )
  : Query(trie,data,timeLimit,continuous, error_callback, fast_error_callback, predictor_error_callback, maxError, stopAfterError),experror ( experror )
{
}


Query_Best::~Query_Best()
{
}


string Query_Best::printResult ( DataManager *data2 ) {
    return printResult ( data2, (QueryData_Best*) realroot->data );
}

string Query_Best::printResult ( DataManager *data2, QueryData_Best *data ) {
    int depth;
    string out = "";
    out += "(nItems, nTransactions) : ( " + std::to_string(data2->getNAttributes()*2) + ", " + std::to_string(data2->getNTransactions()) + " )\n";
    out += "Tree: ";
    if ( data->size == 0 || (data->size == 1 && data->error == FLT_MAX) ){
        out += "(No such tree)\n";
        printTimeOut(&out);
        return out;
    }
    else {
        depth = printResult ( data, 1, &out );
        out += "}\n";
        out += "Size: " + std::to_string(data->size) + "\n";
        out += "Depth: " + std::to_string(depth - 1) + "\n";
        out += "Error: " + std::to_string(data->error) + "\n";
        printAccuracy(data2, data, &out);
        printTimeOut(&out);
        return out;
    }
}

int Query_Best::printResult ( QueryData_Best *data, int depth, string* out ) {
    if ( data->left == NULL ) { // leaf
        if (predictor_error_callback != nullptr)
            *out += "{\"value\": \"undefined\", \"error\": " + std::to_string(data->error);
        else
            *out += "{\"class\": " + std::to_string(data->test) + ", \"error\": " + std::to_string(data->error);
        return depth;
    }
    else {
        if (continuous)
            *out += "{\"feat\": " + ((DataContinuous*) this->data)->names[data->test] + ", \"left\": ";
        else
            *out += "{\"feat\": " + std::to_string(data->test) + ", \"left\": ";
        int d1 = printResult ( data->right, depth + 1, out );
        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        *out += "}, \"right\": ";
        int d2 = printResult ( data->left, depth + 1, out );
        *out += "}";
        return max ( d1, d2 );
    }
}

void Query_Best::printTimeOut(string* out){
    if (timeLimitReached)
        *out += "Timeout\n";// << endl;
}

bool Query_Best::canimprove ( QueryData *left, Error ub ) {
    return ((QueryData_Best*) left )->error < ub ;
}

bool Query_Best::canSkip( QueryData *actualBest ){
    return ((QueryData_Best*) actualBest )->error == ((QueryData_Best*) actualBest )->lowerBound;
}

void Query_Best::printAccuracy ( DataManager *data2, QueryData_Best *data, string* out ) {
    *out += "Accuracy: 0\n";// << endl;
}

/*Class Query_Best::runResult ( DataManager *data, Transaction transaction ) {
  return runResult ( (QueryData_Best*) realroot->data, data, transaction );  
}

Class Query_Best::runResult ( QueryData_Best *data2, DataManager *data, Transaction transaction ) {
  if ( data2->left == NULL )  // leaf
    return data2->test;
  else
    if ( data->isIn ( transaction, data2->test ) )
      return runResult ( data2->right, data, transaction );
    else
      return runResult ( data2->left, data, transaction );
}*/
