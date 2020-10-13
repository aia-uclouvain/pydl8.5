#include "query_weighted.h"
#include "trie.h"
#include <iostream>

Query_Weighted::Query_Weighted(Trie *trie,
                               DataManager *data,
                               ExpError *experror,
                               int timeLimit,
                               bool continuous,
                               function<vector<float>(RCover *)>* tids_error_class_callback,
                               function<vector<float>(RCover *)>* supports_error_class_callback,
                               function<float(RCover *)>* tids_error_callback,
                               function<vector<float>(string)>* example_weight_callback,
                               function<float(string)>* predict_error_callback,
                               float maxError,
                               bool stopAfterError) :
        Query_Best(trie,
                   data,
                   experror,
                   timeLimit,
                   continuous,
                   tids_error_class_callback,
                   supports_error_class_callback,
                   tids_error_callback,
                   maxError,
                   stopAfterError),
        example_weight_callback(example_weight_callback),
        predict_error_callback(predict_error_callback) {}


Query_Weighted::~Query_Weighted() {
    delete example_weight_callback;
    delete predict_error_callback;
}


bool Query_Weighted::is_freq(pair<Supports, Support> supports) {
    return supports.second >= minsup;
}

bool Query_Weighted::is_pure(pair<Supports, Support> supports) {
    Support majnum = supports.first[0], secmajnum = 0;
    for (int i = 1; i < nclasses; ++i)
        if (supports.first[i] > majnum) {
            secmajnum = majnum;
            majnum = supports.first[i];
        } else if (supports.first[i] > secmajnum)
            secmajnum = supports.first[i];
    return ((long int) minsup - (long int) (supports.second - majnum)) > (long int) secmajnum;
}

bool
Query_Weighted::updateData(QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right) {
    QueryData_Best *best2 = (QueryData_Best *) best, *left2 = (QueryData_Best *) left, *right2 = (QueryData_Best *) right;
    Error error = left2->error + right2->error;
    Size size = left2->size + right2->size + 1;
    if (error < upperBound || (error == upperBound && size < best2->size)) {
        best2->error = error;
        best2->left = left2;
        best2->right = right2;
        best2->size = size;
        best2->test = attribute;
        return true;
    }
    return false;
}

QueryData *Query_Weighted::initData(RCover *cover, Depth currentMaxDepth) {
    Class maxclass = -1;
    Error error;

    QueryData_Best *data = new QueryData_Best();
    if (tids_error_class_callback == nullptr &&
        tids_error_callback == nullptr) { //fast or default error. support will be used
        if (supports_error_class_callback != nullptr) { //python fast error
            function<vector<float>(RCover *)> callback = *supports_error_class_callback;
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        } else { //default error
            ErrorValues ev = computeErrorValues(cover);
            error = ev.error;
            maxclass = ev.maxclass;
        }
    } else { //slow error or predictor error function. Not need to compute support
        if (tids_error_callback != nullptr) {
            function<float(RCover *)> callback = *tids_error_callback;
            error = callback(cover);
        } else {
            function<vector<float>(RCover *)> callback = *tids_error_class_callback;
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        }
    }
    data->test = maxclass;
    data->leafError = error;
    data->error += experror->addError(cover->getSupport(), data->error, dm->getNTransactions());
    data->solutionDepth = currentMaxDepth;
//    if (conflict > 0) data2->right = (QueryData_Best *) 1;

    return (QueryData *) data;
}

ErrorValues Query_Weighted::computeErrorValues(RCover *cover) {
    Class maxclass;
    Error error;

    Supports itemsetSupport = cover->getWeightedSupportPerClass(weights);
    Support maxclassval = itemsetSupport[0];
    maxclass = 0;

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (itemsetSupport[i] == maxclassval) {
            if (dm->getSupports()[i] > dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = cover->getSupport() - maxclassval;
    return {error, maxclass};
}


ErrorValues Query_Weighted::computeErrorValues(Supports itemsetSupport, bool onlyerror) {
    Class maxclass = 0;
    Error error;
    Support maxclassval = itemsetSupport[0];

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (itemsetSupport[i] == maxclassval) {
            if (dm->getSupports()[i] > dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;

    return {error, maxclass};
}


Error Query_Weighted::computeOnlyError(Supports itemsetSupport) {
    Class maxclass = 0;
    Support maxclassval = itemsetSupport[0];

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (itemsetSupport[i] == maxclassval) {
            if (dm->getSupports()[i] > dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    return sumSupports(itemsetSupport) - maxclassval;
}

Error Query_Weighted::getTrainingError(const string& tree_json){
    function<float(string)> callback = *predict_error_callback;
    return callback(tree_json);
}
