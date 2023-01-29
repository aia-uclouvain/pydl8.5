#include "nodeDataManager.h"

NodeDataManager::NodeDataManager(
        RCover* cover,
             function<vector<float>(RCover *)> *tids_error_class_callback,
             function<vector<float>(RCover *)> *supports_error_class_callback,
             function<float(RCover *)> *tids_error_callback
             ) : cover(cover), tids_error_class_callback(tids_error_class_callback),
                                    supports_error_class_callback(supports_error_class_callback),
                                    tids_error_callback(tids_error_callback)
{}


NodeDataManager::~NodeDataManager() {
}


LeafInfo NodeDataManager::computeLeafInfo(RCover *cov) {
    if (cov == nullptr) cov = cover;
    Class maxclass;
    Error error;
    ErrorVals itemsetSupport = cov->getErrorValPerClass();
    ErrorVal maxclassval = itemsetSupport[0];
    maxclass = 0;

    for (int i = 1; i < GlobalParams::getInstance()->nclasses; ++i) {
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


LeafInfo NodeDataManager::computeLeafInfo(ErrorVals itemsetSupport) {
    Class maxclass = 0;
    Error error;
    ErrorVal maxclassval = itemsetSupport[0];

    for (int i = 1; i < GlobalParams::getInstance()->nclasses; ++i) {
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


