#include "query_totalfreq.h"
#include "trie.h"
#include <iostream>
#include <stdlib.h>

Query_TotalFreq::Query_TotalFreq(Trie *trie, Data *data, ExpError *experror, int timeLimit, bool continuous, float maxError, bool stopAfterError )
        : Query_Best(trie,data,experror,timeLimit,continuous, maxError, stopAfterError) {
}


Query_TotalFreq::~Query_TotalFreq() {
}


bool Query_TotalFreq::is_freq ( pair<Supports,Support> supports ) {
    //cout << "support = " << supports.second << " et minsup = " << minsup << endl;
    return supports.second >= minsup;
}

bool Query_TotalFreq::is_pure ( pair<Supports,Support> supports ) {
    Support majnum = supports.first[0], secmajnum = 0;
    for ( int i = 1; i < nclasses; ++i )
        if ( supports.first[i] > majnum ) {
            secmajnum = majnum;
            majnum = supports.first[i];
        }
        else
        if ( supports.first[i] > secmajnum )
            secmajnum = supports.first[i];
    return ( (long int) minsup - (long int) ( supports.second - majnum ) ) > (long int) secmajnum;
}

bool Query_TotalFreq::updateData ( QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right) {
    QueryData_Best *best2 = (QueryData_Best*) best,
            *left2 = (QueryData_Best*) left,
            *right2 = (QueryData_Best*) right;
    Error error = left2->error + right2->error;
    Size size = left2->size + right2->size + 1;
    if ( error <= upperBound ||
         ( error == upperBound && size < best2->size ) ) {
        best2->error = error;
        best2->left = left2;
        best2->right = right2;
        best2->size = size;
        best2->test = attribute;
        return true;
    }
    return false;
}

QueryData *Query_TotalFreq::initData ( pair<Supports,Support> supports, Error parent_ub, Support minsup, Depth currentMaxDepth ) {
    Support maxclass = 0, maxclassval = supports.first[0], minclassval = supports.first[0];
    //cout << "tot freq" << endl;
    int conflict = 0;
    for ( int i = 1; i < nclasses; ++i )
        if ( supports.first[i] > maxclassval ) {
            maxclassval = supports.first[i];
            maxclass = i;
            conflict = 0;
        }
        else
        if ( supports.first[i] == maxclassval ) {
            ++conflict; // two with the same label
            if ( data->getSupports() [i] > data->getSupports() [maxclass] )
                maxclass = i;
        } else
            minclassval = supports.first[i];
    //QueryData_Best *data2 = (QueryData_Best*) malloc(sizeof(QueryData_Best));
    QueryData_Best *data2 = new QueryData_Best();
    data2->test = maxclass;
    data2->left = data2->right = NULL;
    data2->leafError = supports.second - maxclassval;
    data2->error = FLT_MAX;
    data2->error += experror->addError ( supports.second, data2->error, data->getNTransactions() );
    data2->size = 1;
    data2->initUb = min(parent_ub, data2->leafError );
    data2->solutionDepth = currentMaxDepth;
    data2->nTransactions = supports.second;
    if ( conflict )
        data2->right = (QueryData_Best*) 1;
    if ( minsup <= minclassval )
        data2->lowerBound = 0;
    else{
        if (nclasses == 2){
            data2->lowerBound = min(minclassval, minsup-minclassval);// minsup - maxclassval;
        }
        else{
            //cout << "bingo" << endl;
            data2->lowerBound = 0;
        }

    }

    return (QueryData*) data2;
}

void Query_TotalFreq::printAccuracy ( Data *data2, QueryData_Best *data, string* out ) {
    //cout << "Accuracy: " << (data2->getNTransactions() - data->error) / (double) data2->getNTransactions() << endl;
    *out += "Accuracy: " + std::to_string((data2->getNTransactions() - data->error) / (double) data2->getNTransactions()) + "\n";
}

