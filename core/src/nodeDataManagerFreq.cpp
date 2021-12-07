#include "nodeDataManagerFreq.h"
#include "cache.h"
#include <iostream>
#include <stack>
#include "logger.h"

NodeDataManagerFreq::NodeDataManagerFreq(
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
                   {}


NodeDataManagerFreq::~NodeDataManagerFreq() {}


bool NodeDataManagerFreq::updateData(Node *best, Error upperBound, Attribute attribute, Node *left, Node *right, Cache* cache) {
    auto *freq_best = (Freq_NodeData *) best->data, *freq_left = (Freq_NodeData *) left->data, *freq_right = (Freq_NodeData *) right->data;
    Error error = freq_left->error + freq_right->error;
    Size size = freq_left->size + freq_right->size + 1;
    if (error < upperBound || (floatEqual(error, upperBound) && size < freq_best->size)) {
        freq_best->error = error;
        freq_best->left = freq_left;
        freq_best->right = freq_right;
        freq_best->size = size;
        freq_best->test = attribute;

//        cache->updateParents(best, left, right);

        return true;
    }
    return false;
}

NodeData *NodeDataManagerFreq::initData(RCover *cov, Depth currentMaxDepth, int hashcode) {
    Class maxclass = -1;
//    float maxclass = -1;
    Error error;

    auto *data = new Freq_NodeData();

    if (cov == nullptr) cov = cover;

//    if (trie->use_priority) trie->nodemapper.push({cover->getSupport(), hashcode});
//    cout << "fi1" << endl;

    //fast or default error. support will be used
    if (tids_error_class_callback == nullptr && tids_error_callback == nullptr) {
//        cout << "fi2" << endl;
        //python fast error
        if (supports_error_class_callback != nullptr) {
//            cout << "fi3" << endl;
            function<vector<float>(RCover *)> callback = *supports_error_class_callback;
            cov->getErrorValPerClass(); // allocate the sup_array if it does not exist yet and compute the frequency counts
            vector<float> infos = callback(cover);
            error = infos[0];
//            maxclass = infos[1];
            maxclass = int(infos[1]);
        }
        //default error
        else {
//            cout << "fi4" << endl;
            LeafInfo ev = computeLeafInfo();
            error = ev.error;
            maxclass = ev.maxclass;
        }
    }
    //slow error or predictor error function. Not need to compute support
    else {
//        cout << "fi5" << endl;
        if (tids_error_callback != nullptr) {
//            cout << "fi6" << endl;
            function<float(RCover *)> callback = *tids_error_callback;
            error = callback(cov);
        } else {
//            cout << "fi7" << endl;
            function<vector<float>(RCover *)> callback = *tids_error_class_callback;
            vector<float> infos = callback(cov);
            error = infos[0];
//            maxclass = infos[1];
            maxclass = int(infos[1]);
//            cout << infos[0] << " " << infos[1] << endl;
//            cout << error << " " << maxclass << endl;
        }
    }
    data->test = maxclass;
    data->leafError = error;
//    data->error += experror->addError(cover->getSupport(), data->error, dm->getNTransactions());
//    data->solutionDepth = currentMaxDepth;

    return (NodeData *) data;
}

LeafInfo NodeDataManagerFreq::computeLeafInfo(RCover *cov) {
    if (cov == nullptr) cov = cover;
    Class maxclass;
    Error error;
    ErrorVals itemsetSupport = cov->getErrorValPerClass();
    ErrorVal maxclassval = itemsetSupport[0];
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
    error = sumErrorVals(itemsetSupport) - maxclassval;
    return {error, maxclass};
}


LeafInfo NodeDataManagerFreq::computeLeafInfo(ErrorVals itemsetSupport) {
    Class maxclass = 0;
    Error error;
    ErrorVal maxclassval = itemsetSupport[0];

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (floatEqual(itemsetSupport[i], maxclassval)) {
            if (cover->dm->getSupports()[i] > cover->dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumErrorVals(itemsetSupport) - maxclassval;
    return {error, maxclass};
}
