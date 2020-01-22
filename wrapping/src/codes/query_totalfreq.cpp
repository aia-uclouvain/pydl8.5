#include "query_totalfreq.h"
#include "trie.h"
#include <iostream>

Query_TotalFreq::Query_TotalFreq(Trie *trie,DataManager *data, ExpError *experror, int timeLimit, bool continuous,
                                 function<vector<float>(RCover * )> *error_callback,
                                 function<vector<float>(RCover * )> *fast_error_callback,
                                 function<float(RCover * )> *predictor_error_callback, float maxError,
                                 bool stopAfterError)
                                : Query_Best(trie, data, experror, timeLimit, continuous, error_callback,
                                        fast_error_callback, predictor_error_callback,
                                        maxError, stopAfterError) {}


Query_TotalFreq::~Query_TotalFreq() {}


bool Query_TotalFreq::is_freq(pair <Supports, Support> supports) {
    return supports.second >= minsup;
}

bool Query_TotalFreq::is_pure(pair <Supports, Support> supports) {
    Support majnum = supports.first[0], secmajnum = 0;
    for (int i = 1; i < nclasses; ++i)
        if (supports.first[i] > majnum) {
            secmajnum = majnum;
            majnum = supports.first[i];
        } else if (supports.first[i] > secmajnum)
            secmajnum = supports.first[i];
    return ((long int) minsup - (long int) (supports.second - majnum)) > (long int) secmajnum;
}

bool Query_TotalFreq::updateData(QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right) {
    QueryData_Best *best2 = (QueryData_Best *) best, *left2 = (QueryData_Best *) left, *right2 = (QueryData_Best *) right;
    Error error = left2->error + right2->error;
    Size size = left2->size + right2->size + 1;
    if (error < upperBound) {
        best2->error = error;
        best2->left = left2;
        best2->right = right2;
        best2->size = size;
        best2->test = attribute;
        return true;
    }
    else if (error == upperBound){
        if (best2->error == FLT_MAX)
            best2->error = error;
        /*else if (size < best2->size){
            best2->error = error;
            best2->left = left2;
            best2->right = right2;
            best2->size = size;
            best2->test = attribute;
        }*/
    }
    return false;
}

QueryData *Query_TotalFreq::initData(RCover *cover, Error parent_ub, Support minsup, Depth currentMaxDepth) {

    pair <Supports, Support> itemsetSupport;//declare variable of pair type to keep firstly an array of support per class and second the support of the itemset
    Class maxclass = -1;
    Error error;
    int conflict = 0;
    Error lowerb = 0;

    if (error_callback == nullptr && predictor_error_callback == nullptr) {//fast or default error. support will be used
        itemsetSupport = cover->getSupportPerClass();
        cover->sup = itemsetSupport.first;

        if (fast_error_callback != nullptr) {//python fast error
            function < vector<float>(RCover * ) > callback = *fast_error_callback;
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        } else {//default error
            Support maxclassval = itemsetSupport.first[0];
            maxclass = 0;
            int secondval = -1;
            for (int i = 1; i < nclasses; ++i) {
                if (itemsetSupport.first[i] > maxclassval) {
                    if (maxclassval > secondval)
                        secondval = maxclassval;
                    maxclassval = itemsetSupport.first[i];
                    maxclass = i;
                    conflict = 0;
                } else if (itemsetSupport.first[i] == maxclassval) {
                    secondval = maxclassval;
                    ++conflict; // two with the same label
                    if (data->getSupports()[i] > data->getSupports()[maxclass])
                        maxclass = i;
                } else{
                    if (itemsetSupport.first[i] > secondval)
                        secondval = itemsetSupport.first[i];
                }

            }
            error = itemsetSupport.second - maxclassval;
            int remaining = itemsetSupport.second - (maxclassval + secondval);
            if (maxclassval >= minsup){
                if (secondval >= minsup)
                    lowerb = remaining;
                else
                    if (secondval + remaining >= minsup)
                        lowerb = remaining;
                    else
                        lowerb = minsup - secondval;
            } else
                if (secondval < minsup)
                    lowerb = remaining;
            lowerb = 0;
        }
        deleteSupports(itemsetSupport.first);
    } else {//slow error or predictor error function. Not need to compute support

        if (predictor_error_callback != nullptr) {
            function<float(RCover * )> callback = *predictor_error_callback;
            error = callback(cover);
        } else {
            function < vector<float>(RCover * ) > callback = *error_callback;
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        }
    }

    QueryData_Best *data2 = new QueryData_Best();
    data2->test = maxclass;
    data2->left = data2->right = NULL;
    data2->leafError = error;
    data2->error = FLT_MAX;
    data2->error += experror->addError(cover->getSupport(), data2->error, data->getNTransactions());
    data2->size = 1;
    data2->initUb = min(parent_ub, data2->leafError);
    data2->solutionDepth = currentMaxDepth;
    if (conflict > 0)
        data2->right = (QueryData_Best *) 1;
//    if (minclassval != -1 && nclasses == 2 && minsup > minclassval)
//        lowerb = min(minclassval, minsup - minclassval);// minsup - maxclassval;
    data2->lowerBound = lowerb;

    return (QueryData *) data2;
}


void Query_TotalFreq::printAccuracy(DataManager *data2, QueryData_Best *data, string *out) {
    *out += "Accuracy: " + std::to_string((data2->getNTransactions() - data->error) / (double) data2->getNTransactions()) + "\n";
}
