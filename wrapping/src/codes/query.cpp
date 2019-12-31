#include "query.h"
#include <climits>
#include <cfloat>

Query::Query( Trie *trie, DataManager *data, int timeLimit, bool continuous, function<vector<float>(RCover*)>* error_callback, function<vector<float>(RCover*)>* fast_error_callback, function<float(RCover*)>*  predictor_error_callback, float maxError, bool stopAfterError ): trie ( trie ), data ( data ), maxdepth ( NO_ITEM ), timeLimit( timeLimit ), error_callback(error_callback), fast_error_callback(fast_error_callback), predictor_error_callback(predictor_error_callback), maxError(maxError), continuous( continuous ), stopAfterError(stopAfterError)
{
}


Query::~Query()
{
}


