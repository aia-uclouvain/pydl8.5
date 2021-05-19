#include "nodedataManager.h"
#include <climits>
#include <cfloat>

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


