#include "query.h"
#include <climits>
#include <cfloat>

Query::Query( Trie *trie, Data *data, int timeLimit, bool continuous, float maxError, bool stopAfterError ): trie ( trie ), data ( data ), maxdepth ( NO_ITEM ), timeLimit( timeLimit ), maxError(maxError), continuous( continuous ), stopAfterError(stopAfterError)
{
}


Query::~Query()
{
}


