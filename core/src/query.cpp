#include "query.h"
#include <climits>
#include <cfloat>

Query::Query(Support minsup,
             Depth maxdepth,
             Trie *trie,
             DataManager *dm,
             int timeLimit,
             function<vector<float>(RCover *)> *tids_error_class_callback,
             function<vector<float>(RCover *)> *supports_error_class_callback,
             function<float(RCover *)> *tids_error_callback,
             float maxError,
             bool stopAfterError) : dm(dm),
                                    trie(trie),
                                    minsup(minsup),
                                    maxdepth(maxdepth),
                                    timeLimit(timeLimit),
                                    maxError(maxError),
                                    stopAfterError(stopAfterError),
                                    tids_error_class_callback(tids_error_class_callback),
                                    supports_error_class_callback(supports_error_class_callback),
                                    tids_error_callback(tids_error_callback)
{}


Query::~Query() {
}


