#include "cache.h"

using namespace std;


Cache::Cache(Depth maxdepth, WipeType wipe_type, Size maxcachesize): wipe_type(wipe_type), maxdepth(maxdepth), maxcachesize(maxcachesize) {
    cachesize = 0;
}



