#include "query.h"
#include <climits>
#include <cfloat>

Query::Query( Trie *trie, Data *data, int timeLimit, bool continuous, function<float(Array<int>*)>* error_callback, function<vector<float>(Array<int>*)>* fast_error_callback, float maxError, bool stopAfterError ): trie ( trie ), data ( data ), maxdepth ( NO_ITEM ), timeLimit( timeLimit ), error_callback(error_callback), fast_error_callback(fast_error_callback), maxError(maxError), continuous( continuous ), stopAfterError(stopAfterError)
{
}


Query::~Query()
{
}


