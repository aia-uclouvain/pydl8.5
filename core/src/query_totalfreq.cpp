#include "query_totalfreq.h"
#include "trie.h"
#include <iostream>

Query_TotalFreq::Query_TotalFreq(Support minsup,
                                 Depth maxdepth,
                                 Trie *trie,
                                 DataManager *data,
                                 int timeLimit,
                                 function<vector<float>(RCover *)> *tids_error_class_callback,
                                 function<vector<float>(RCover *)> *supports_error_class_callback,
                                 function<float(RCover *)> *tids_error_callback,
                                 float* maxError, 
                                 bool* stopAfterError) :
        Query_Best(minsup,
                   maxdepth,
                   trie,
                   data,
                   timeLimit,
                   tids_error_class_callback,
                   supports_error_class_callback,
                   tids_error_callback,
                   maxError,
                   stopAfterError) {

                       if (maxError == nullptr) {
                           maxError = new float[data->getNQuantiles()];
                           stopAfterError = new bool[data->getNQuantiles()];
                           for (int i = 0; i < data->getNQuantiles(); i++) {
                               maxError[i] == NO_ERR;
                               stopAfterError[i] = false;
                           }
                       } else {
                           for (int i = 0; i < data->getNQuantiles(); i++) {
                               maxError[i] = maxError[i] <= 0 ? NO_ERR : maxError[i];
                               stopAfterError[i] = maxError[i] <= 0 ? false : stopAfterError[i];
                           }
                       }

                       quantileLossComputer = new QuantileLossComputer(data->getNQuantiles());
                   }


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

bool Query_TotalFreq::updateData(QueryData *best, Error* upperBound, Attribute attribute, QueryData *left, QueryData *right, Error* minlb) {
    QueryData_Best *best2 = (QueryData_Best *) best, *left2 = (QueryData_Best *) left, *right2 = (QueryData_Best *) right;
    Error error;
    Size size;

    bool changed = false;

    for (int i = 0; i < dm->getNQuantiles(); i++) {
        error = left2->errors[i] + right2->errors[i];
        size = left2->sizes[i] + right2->sizes[i] + 1;

        // TODO VALENTIN : check if this is correct
        if ((error < upperBound[i])  || (floatEqual(error, upperBound[i]) && size < best2->sizes[i])) {
            best2->errors[i] = error;
            best2->lefts[i] = left2;
            best2->rights[i] = right2;
            best2->sizes[i] = size;
            best2->tests[i] = attribute;

            upperBound[i] = error;
            changed = true;
        } else {
            minlb[i] = fmin(minlb[i], error);
        }
    }
    
    return changed;
}

QueryData *Query_TotalFreq::initData(RCover *cover, Depth currentMaxDepth) {
    Class maxclass = -1;
    Error error;
    Error * errors = nullptr;

    auto *data = new QueryData_Best(dm->getNQuantiles());

    //fast or default error. support will be used
    if (tids_error_class_callback == nullptr && tids_error_callback == nullptr) {
        //python fast error
        if (supports_error_class_callback != nullptr) {
            function<vector<float>(RCover *)> callback = *supports_error_class_callback;
            cover->getSupportPerClass(); // allocate the sup_array if it does not exist yet and compute the frequency counts
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        }
        // backup error
        else {
            if (dm->getBackupError() == MISCLASSIFICATION_ERROR) {
                LeafInfo ev = computeLeafInfo(cover);
                error = ev.error;
                maxclass = ev.maxclass;
            } else if (dm->getBackupError() == MSE_ERROR) {
                error = sse_tids_error(cover);
            } else if (dm->getBackupError() == QUANTILE_ERROR) {
                errors = quantileLossComputer->quantile_tids_errors(cover);
            }
        }
    }
    //slow error or predictor error function. Not need to compute support
    else {
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

    if (errors) {
        for (int i = 0; i < dm->getNQuantiles(); i++) {
            data->errors[i] += errors[i];
            data->leafErrors[i] = errors[i];
        }

        delete[] errors;
    } else {
        data->errors[0] += error;
        data->leafErrors[0] = error;
        data->tests[0] = maxclass;
    }

    return (QueryData *) data;
}

LeafInfo Query_TotalFreq::computeLeafInfo(RCover *cover) {
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
    return {error, maxclass};
}


LeafInfo Query_TotalFreq::computeLeafInfo(Supports itemsetSupport) {
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
    return {error, maxclass};
}
