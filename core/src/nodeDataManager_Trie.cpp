#include "nodeDataManager_Trie.h"
#include "cache_trie.h"

NodeDataManager_Trie::NodeDataManager_Trie(
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


NodeDataManager_Trie::~NodeDataManager_Trie() {}


bool NodeDataManager_Trie::updateData(Node *best, Error upperBound, Attribute attribute, Node *left, Node *right, Itemset itemset) {
    auto *freq_best = best->data, *freq_left = left->data, *freq_right = right->data;
    Error error = freq_left->error + freq_right->error;
    Size size = freq_left->size + freq_right->size + 1;
    if (error < upperBound || (floatEqual(error, upperBound) && size < freq_best->size)) {
        freq_best->error = error;
        freq_best->size = size;
        freq_best->test = attribute;
        return true;
    }
    return false;
}

NodeData *NodeDataManager_Trie::initData(RCover *cov, Depth currentMaxDepth, int hashcode) {
    Class maxclass = -1;
    Error error;
    auto *data = new TrieNodeData();
    if (cov == nullptr) cov = cover;

    //fast or default error. support will be used
    if (tids_error_class_callback == nullptr && tids_error_callback == nullptr) {
        //python fast error
        if (supports_error_class_callback != nullptr) {
            function<vector<float>(RCover *)> callback = *supports_error_class_callback;
            cov->getErrorValPerClass(); // allocate the sup_array if it does not exist yet and compute the frequency counts
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

    // the variable `test` is used to save the feature to split on branch nodes and classes at leaf nodes
    // to make them distinguishable, the feature value is positive and classes are transformed by f(x) = -(x+1)
    // attention to recover the right value when printing the tree
    data->test = -(maxclass + 1);
    data->leafError = error;
    return data;
}
