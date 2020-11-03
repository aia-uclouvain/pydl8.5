#include "query.h"
#include <climits>
#include <cfloat>

Query::Query(Support minsup,
             Depth maxdepth,
             Trie *trie,
             DataManager *dm,
             int timeLimit,
             bool continuous,
             function<vector<float>(RCover *)> *tids_error_class_callback,
             function<vector<float>(RCover *)> *supports_error_class_callback,
             function<float(RCover *)> *tids_error_callback,
             function<vector<float>()> *example_weight_callback,
             function<vector<float>(string)> *predict_error_callback,
             float maxError,
             bool stopAfterError) : minsup(minsup),
                                    maxdepth(maxdepth),
                                    dm(dm),
                                    trie(trie),
                                    timeLimit(timeLimit),
                                    continuous(continuous),
                                    maxError(maxError),
                                    stopAfterError(stopAfterError),
                                    tids_error_class_callback(tids_error_class_callback),
                                    supports_error_class_callback(supports_error_class_callback),
                                    tids_error_callback(tids_error_callback),
                                    example_weight_callback(example_weight_callback),
                                    predict_error_callback(predict_error_callback)//,
{}


Query::~Query() {
}


