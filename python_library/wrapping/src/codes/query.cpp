#include "query.h"
#include <climits>
#include <cfloat>

Query::Query(Trie *trie,
             DataManager *dm,
             int timeLimit,
             bool continuous,
             function<vector<float>(RCover *)> *tids_error_class_callback,
             function<vector<float>(RCover *)> *supports_error_class_callback,
             function<float(RCover *)> *tids_error_callback,
             function<vector<float>(string)> *example_weight_callback,
             function<float(string)> *predict_error_callback,
             vector<float> *weights,
             float maxError,
             bool stopAfterError) : dm(dm),
                                    trie(trie),
                                    maxdepth(NO_ITEM),
                                    timeLimit(timeLimit),
                                    continuous(continuous),
                                    maxError(maxError),
                                    stopAfterError(stopAfterError),
                                    tids_error_class_callback(tids_error_class_callback),
                                    supports_error_class_callback(supports_error_class_callback),
                                    tids_error_callback(tids_error_callback),
                                    example_weight_callback(example_weight_callback),
                                    predict_error_callback(predict_error_callback),
                                    weights(weights) {
}


Query::~Query() {
}


