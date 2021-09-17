#include "freq_nodedataManager.h"
#include "cache.h"
#include <iostream>

Freq_NodeDataManager::Freq_NodeDataManager(
        RCover* cover,
                                 function<vector<float>(RCover *)> *tids_error_class_callback,
                                 function<vector<float>(RCover *)> *supports_error_class_callback,
                                 function<float(RCover *)> *tids_error_callback
                                 ) :
        NodeDataManager(
                cover,
        tids_error_class_callback,
        supports_error_class_callback,
        tids_error_callback)
        /*Query_Best(minsup,
                   maxdepth,
                   trie,
                   data,
                   experror,
                   timeLimit,
                   continuous,
                   tids_error_class_callback,
                   supports_error_class_callback,
                   tids_error_callback,
                   (maxError <= 0) ? NO_ERR : maxError,
                   (maxError <= 0) ? false : stopAfterError) */
                   {}


Freq_NodeDataManager::~Freq_NodeDataManager() {}


//bool Frequency_NodeDataManager::is_freq(pair<Supports, Support> supports) {
//    return supports.second >= minsup;
//}
//
//bool Frequency_NodeDataManager::is_pure(pair<Supports, Support> supports) {
//    Support majnum = supports.first[0], secmajnum = 0;
//    for (int i = 1; i < nclasses; ++i)
//        if (supports.first[i] > majnum) {
//            secmajnum = majnum;
//            majnum = supports.first[i];
//        } else if (supports.first[i] > secmajnum)
//            secmajnum = supports.first[i];
//    return ((long int) minsup - (long int) (supports.second - majnum)) > (long int) secmajnum;
//}

bool Freq_NodeDataManager::updateData(NodeData *best, Error upperBound, Attribute attribute, NodeData *left, NodeData *right) {
//    cout << "update" << endl;
    Freq_NodeData *best2 = (Freq_NodeData *) best;
//    cout << "update1" << endl;
    Freq_NodeData *left2 = (Freq_NodeData *) left;
//    cout << "update2" << endl;
    Freq_NodeData *right2 = (Freq_NodeData *) right;
//    cout << "update3" << endl;
//    cout << right2->error << endl;
//    cout << left2->error << endl;
    Error error = left2->error + right2->error;
//    cout << "update4" << endl;
    Size size = left2->size + right2->size + 1;
//    cout << "update5" << endl;
    if (error < upperBound || (floatEqual(error, upperBound) && size < best2->size)) {
//        cout << "update6" << endl;
        best2->error = error;
//        cout << "update7" << endl;
        best2->left = left2;
//        cout << "update8" << endl;
        best2->right = right2;
//        cout << "update9" << endl;
        best2->size = size;
//        cout << "update10" << endl;
        best2->test = attribute;
//        cout << "update11" << endl;
        return true;
    }
//    cout << "update11" << endl;
    return false;
}

NodeData *Freq_NodeDataManager::initData(RCover *cov, Depth currentMaxDepth, int hashcode) {
    Class maxclass = -1;
    Error error;

    auto *data = new Freq_NodeData();

    if (cov == nullptr) cov = cover;

//    if (trie->use_priority) trie->nodemapper.push({cover->getSupport(), hashcode});

    //fast or default error. support will be used
    if (tids_error_class_callback == nullptr && tids_error_callback == nullptr) {
        //python fast error
        if (supports_error_class_callback != nullptr) {
            function<vector<float>(RCover *)> callback = *supports_error_class_callback;
            cov->getSupportPerClass(); // allocate the sup_array if it does not exist yet and compute the frequency counts
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        }
        //default error
        else {
            LeafInfo ev = computeLeafInfo();
            error = ev.error;
            maxclass = ev.maxclass;
        }
    }
    //slow error or predictor error function. Not need to compute support
    else {
        if (tids_error_callback != nullptr) {
            function<float(RCover *)> callback = *tids_error_callback;
            error = callback(cov);
        } else {
            function<vector<float>(RCover *)> callback = *tids_error_class_callback;
            vector<float> infos = callback(cov);
            error = infos[0];
            maxclass = int(infos[1]);
        }
    }
    data->test = maxclass;
    data->leafError = error;
//    data->error += experror->addError(cover->getSupport(), data->error, dm->getNTransactions());
    data->solutionDepth = currentMaxDepth;

    return (NodeData *) data;
}

LeafInfo Freq_NodeDataManager::computeLeafInfo(RCover *cov) {
    if (cov == nullptr) cov = cover;
    Class maxclass;
    Error error;
    Supports itemsetSupport = cov->getSupportPerClass();
    SupportClass maxclassval = itemsetSupport[0];
    maxclass = 0;

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (floatEqual(itemsetSupport[i], maxclassval)) {
            if (cov->dm->getSupports()[i] > cov->dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;
    return {error, maxclass};
}


LeafInfo Freq_NodeDataManager::computeLeafInfo(Supports itemsetSupport) {
    Class maxclass = 0;
    Error error;
    SupportClass maxclassval = itemsetSupport[0];

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (floatEqual(itemsetSupport[i], maxclassval)) {
            if (cover->dm->getSupports()[i] > cover->dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;
    return {error, maxclass};
}
