#include "query_totalfreq.h"
#include "trie.h"
#include <iostream>

Query_TotalFreq::Query_TotalFreq(Trie *trie,DataManager *data, ExpError *experror, int timeLimit, bool continuous,
                                 function<vector<float>(RCover* )> *error_callback,
                                 function<vector<float>(RCover* )> *fast_error_callback,
                                 function<float(RCover* )> *predictor_error_callback,
                                 float maxError, bool stopAfterError)
                                : Query_Best(trie, data, experror, timeLimit, continuous,
                                        error_callback, fast_error_callback,
                                        predictor_error_callback, maxError, stopAfterError) {}


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
    if (error < upperBound || (error == upperBound && size < best2->size)) {
        best2->error = error;
        best2->left = left2;
        best2->right = right2;
        best2->size = size;
        best2->test = attribute;
        plusSupports(left2->corrects, right2->corrects, best2->corrects);
        plusSupports(left2->falses, right2->falses, best2->falses);
        return true;
    }
    return false;
}

QueryData *Query_TotalFreq::initData(RCover *cover, Depth currentMaxDepth) {
    Class maxclass = -1, conflict = 0;
    Error error, lowerb = 0;
    Supports corrects = nullptr, falses = nullptr;

    QueryData_Best *data2 = new QueryData_Best();
    if (error_callback == nullptr && predictor_error_callback == nullptr) { //fast or default error. support will be used
        if (fast_error_callback != nullptr) { //python fast error
            function < vector<float>(RCover*)> callback = *fast_error_callback;
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        } else { //default error
            ErrorValues ev = computeErrorValues(cover);
            error = ev.error;
            lowerb = ev.lowerb;
            maxclass = ev.maxclass;
            conflict = ev.conflict;
            corrects = ev.corrects;
            falses = ev.falses;
        }
    } else { //slow error or predictor error function. Not need to compute support
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
    data2->test = maxclass;
    data2->left = data2->right = NULL;
    data2->leafError = error;
    data2->error = FLT_MAX;
    data2->error += experror->addError(cover->getSupport(), data2->error, data->getNTransactions());
    data2->size = 1;
    data2->corrects = corrects;
    data2->falses = falses;
    data2->solutionDepth = currentMaxDepth;
    if (conflict > 0) data2->right = (QueryData_Best *) 1;
    data2->lowerBound = lowerb;

    return (QueryData *) data2;
}

ErrorValues Query_TotalFreq::computeErrorValues(RCover* cover) {
    Class maxclass;
    Error error;
    int conflict = 0;
    Error lowerb = 0;
    Supports corrects = zeroSupports(), falses = zeroSupports();

    Supports itemsetSupport = cover->getSupportPerClass();
    Support maxclassval = itemsetSupport[0];
    maxclass = 0;

    if (corrects && falses) corrects[0] = itemsetSupport[0];
    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            //if (corrects && falses){
                falses[maxclass] = corrects[maxclass];
                corrects[maxclass] = 0;
                corrects[i] = maxclassval;
            //}
            maxclass = i;
            conflict = 0;
        } else if (itemsetSupport[i] == maxclassval) {
            ++conflict; // two with the same label
            //if (corrects && falses){
                falses[maxclass] = corrects[maxclass];
                corrects[maxclass] = 0;
                corrects[i] = maxclassval;
            //}
            if (data->getSupports()[i] > data->getSupports()[maxclass])
                maxclass = i;
        }

    }
    error = cover->getSupport() - maxclassval;

    return {error, lowerb, maxclass, conflict, corrects, falses};
}


ErrorValues Query_TotalFreq::computeErrorValues(Supports itemsetSupport, bool onlyerror) {
    Class maxclass = 0;
    Error error;
    Support maxclassval = itemsetSupport[0];
    Supports corrects = nullptr, falses = nullptr;
    if (!onlyerror) {
        corrects = zeroSupports(), falses = zeroSupports();
        corrects[0] = maxclassval;
    }

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            if (!onlyerror){
                falses[maxclass] = corrects[maxclass];
                corrects[maxclass] = 0;
                corrects[i] = maxclassval;
            }
            maxclass = i;
        } else if (itemsetSupport[i] == maxclassval) {
            if (!onlyerror){
                falses[maxclass] = corrects[maxclass];
                corrects[maxclass] = 0;
                corrects[i] = maxclassval;
            }
            if (data->getSupports()[i] > data->getSupports()[maxclass])
                maxclass = i;
        } else{
            falses[i] = itemsetSupport[i];
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;

    //return {error, 0, maxclass, 0, nullptr, nullptr};

    if (onlyerror) return {error, 0, maxclass, 0, nullptr, nullptr};
    else {
        //cout << "ffff " << falses[0] << "  " << falses[1] << endl;
        return {error, 0, maxclass, 0, corrects, falses};
    }
}


void Query_TotalFreq::printAccuracy(DataManager *data2, QueryData_Best *data, string *out) {
    *out += "Accuracy: " + std::to_string((data2->getNTransactions() - data->error) / (double) data2->getNTransactions()) + "\n";
}
