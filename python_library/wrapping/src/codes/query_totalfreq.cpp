#include "query_totalfreq.h"
#include "trie.h"
#include <iostream>

Query_TotalFreq::Query_TotalFreq(Support minsup,
                                 Depth maxdepth,
                                 Trie *trie,
                                 DataManager *data,
                                 ExpError *experror,
                                 int timeLimit,
                                 bool continuous,
                                 function<vector<float>(RCover *)> *tids_error_class_callback,
                                 function<vector<float>(RCover *)> *supports_error_class_callback,
                                 function<float(RCover *)> *tids_error_callback,
                                 function<vector<float>()> *example_weight_callback,
                                 function<vector<float>(string)> *predict_error_callback,
                                 float maxError, bool stopAfterError) :
        Query_Best(minsup,
                   maxdepth,
                   trie,
                   data,
                   experror,
                   timeLimit,
                   continuous,
                   tids_error_class_callback,
                   supports_error_class_callback,
                   tids_error_callback,
                   example_weight_callback,
                   predict_error_callback,
                   (maxError <= 0) ? NO_ERR : maxError,
                   (maxError <= 0) ? false : stopAfterError) {}


Query_TotalFreq::~Query_TotalFreq() {}


bool Query_TotalFreq::is_freq(pair<Supports, Support> supports) {
    return supports.second >= minsup;
}

bool Query_TotalFreq::is_pure(pair<Supports, Support> supports) {
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
    if (error < upperBound || (floatEqual(error, upperBound) && size < best2->size)) {
        best2->error = error;
        best2->left = left2;
        best2->right = right2;
        best2->size = size;
        best2->test = attribute;
        return true;
    }
    return false;
}

QueryData *Query_TotalFreq::initData(RCover *cover, Depth currentMaxDepth) {
    Class maxclass = -1;
    Error error;
    //cout << "comp err" << endl;

    auto *data = new QueryData_Best();

    //fast or default error. support will be used
    if (tids_error_class_callback == nullptr && tids_error_callback == nullptr) {
        //python fast error
        if (supports_error_class_callback != nullptr) {
            //cout << "seriu" << endl;
            function<vector<float>(RCover *)> callback = *supports_error_class_callback;
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        }
            //default error
        else {
            //cout << "jjoj" << endl;
            ErrorValues ev = computeErrorValues(cover);
            error = ev.error;
            maxclass = ev.maxclass;
        }
    }
    //slow error or predictor error function. Not need to compute support
    else {
        if (tids_error_callback != nullptr) {
            //cout << "bbb" << endl;
            function<float(RCover *)> callback = *tids_error_callback;
            error = callback(cover);
        } else {
            //cout << "devrait appeler" << endl;
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

    return (QueryData *) data;
}

ErrorValues Query_TotalFreq::computeErrorValues(RCover *cover) {
    Class maxclass;
    Error error;

    Supports itemsetSupport = cover->getSupportPerClass();
    SupportClass maxclassval = itemsetSupport[0];
    maxclass = 0;

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (floatEqual(itemsetSupport[i], maxclassval)) {
            if (dm->getSupports()[i] > dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;
//    if (error < 0) cout << "sup[0] = " << itemsetSupport[0] << ", " << "sup[1] = " << itemsetSupport[1] << " sum = " << sumSupports(itemsetSupport) << " maxclassval = " << maxclassval << " error = " << error << " class = " << maxclass << endl;
    return {error, maxclass};
}


ErrorValues Query_TotalFreq::computeErrorValues(Supports itemsetSupport) {
    Class maxclass = 0;
    Error error;
    SupportClass maxclassval = itemsetSupport[0];

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (floatEqual(itemsetSupport[i], maxclassval)) {
            if (dm->getSupports()[i] > dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;
//    if (error < 0) cout << "sup[0] = " << itemsetSupport[0] << ", " << "sup[1] = " << itemsetSupport[1] << " sum = " << sumSupports(itemsetSupport) << " maxclassval = " << maxclassval << " error = " << error << " class = " << maxclass << endl;
    return {error, maxclass};
}
