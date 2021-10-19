#include "cache.h"

using namespace std;


Cache::Cache(Depth maxdepth, WipeType wipe_type, Size maxcachesize, bool with_cache): wipe_type(wipe_type), maxdepth(maxdepth), maxcachesize(maxcachesize), with_cache(with_cache) {
    cachesize = 0;
    if (not with_cache) maxcachesize = NO_CACHE_LIMIT;
}



