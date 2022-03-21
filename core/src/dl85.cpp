#include "dl85.h"

using namespace std::chrono;

//bool verbose = false;

//bool Logger::enable = true;

string launch(ErrorVals supports,
              Transaction ntransactions,
              Attribute nattributes,
              Class nclasses,
              Bool *data,
              Class *target,
              Depth maxdepth,
              Support minsup,
              Error maxError,
              bool stopAfterError,
              function<vector<float>(RCover *)> tids_error_class_callback,
              function<vector<float>(RCover *)> supports_error_class_callback,
              function<float(RCover *)> tids_error_callback,
              float *in_weights,
              bool tids_error_class_is_null,
              bool supports_error_class_is_null,
              bool tids_error_is_null,
              bool infoGain,
              bool infoAsc,
              bool repeatSort,
              int timeLimit,
              bool verbose_param,
              CacheType cache_type,
              int max_cache_size,
              WipeType wipe_type,
              float wipe_factor,
              bool with_cache,
              bool useSpecial,
              bool use_ub,
              bool similarlb,
              bool dynamic_branching,
              bool similar_for_branching,
              bool from_cpp) {

    //as cython can't set null to function, we use a flag to set the appropriated functions to null in c++
    function<vector<float>(RCover *)> *tids_error_class_callback_pointer = &tids_error_class_callback;
    if (tids_error_class_is_null) tids_error_class_callback_pointer = nullptr;

    function<vector<float>(RCover *)> *supports_error_class_callback_pointer = &supports_error_class_callback;
    if (supports_error_class_is_null) supports_error_class_callback_pointer = nullptr;

    function<float(RCover *)> *tids_error_callback_pointer = &tids_error_callback;
    if (tids_error_is_null) tids_error_callback_pointer = nullptr;

    verbose = verbose_param;
//    if (verbose_param) Logger::setTrue();
//    else Logger::setFalse();

    auto *dataReader = new DataManager(supports, ntransactions, nattributes, nclasses, data, target);

    string out = "(nFeats, nTransactions) : ( " + to_string(dataReader->getNAttributes()) + ", " + to_string(dataReader->getNTransactions()) + " )\n";
    out += "Class Distribution: ";
    forEachClass(i) {
        if (i == nclasses - 1) out += to_string(i) + ":" + custom_to_str(dataReader->getSupports()[i]);
        else out += to_string(i) + ":" + custom_to_str(dataReader->getSupports()[i]) + ", ";
    }
    out += "\nmaxdepth: " + to_string(maxdepth) + " --- minsup: " + to_string(minsup) + "\n";

    vector<float> weights;
    if (in_weights) weights = vector<float>(in_weights, in_weights + ntransactions);

    Cache *cache;
    switch (cache_type) {
//        case CacheHash:
//            cache = new Cache_Hash(maxdepth, wipe_type, max_cache_size);
//            //cout << "caching with hashmap limited to " << max_cache_size << " elements" << endl;
//            break;
        case CacheHashCover:
            cache = new Cache_Hash_Cover(maxdepth, wipe_type, max_cache_size, wipe_factor);
            break;
        case CacheHashItemset:
            cache = new Cache_Hash_Itemset(maxdepth, wipe_type, max_cache_size, wipe_factor);
            break;
        default:
            cache = new Cache_Trie(maxdepth, wipe_type, max_cache_size, wipe_factor);
            //cout << "caching with trie limited to " << max_cache_size << " elements" << endl;
            break;
    }

    // use the correct cover depending on whether a weight array is provided or not
    RCover *cover;
    if (in_weights) cover = new RCoverWeight(dataReader, &weights);
    else cover = new RCoverFreq(dataReader); // non-weighted cover

    Search_base *searcher;
    Solution *solution;
    NodeDataManager *nodeDataManager;
    if (with_cache) {
        if (cache_type == CacheHashCover) {
            out += "Storage key: Cover\n";
            if (max_cache_size > NO_CACHE_LIMIT) out += "Cache limitation is not yet supported for cover-based caching\n";
            nodeDataManager = new NodeDataManager_Cover(cover, tids_error_class_callback_pointer, supports_error_class_callback_pointer, tids_error_callback_pointer);
            searcher = new Search_cover_cache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, cache, maxError <= 0 ? NO_ERR : maxError, useSpecial, maxError <= 0 ? false : stopAfterError, similarlb, dynamic_branching, similar_for_branching, from_cpp);
            solution = new Solution_Cover(searcher);
        }
        else {
            if (cache_type == CacheHashItemset) {
                out += "Storage key: Hash Itemset\n";
                if (max_cache_size > NO_CACHE_LIMIT) out += "Cache limitation is not yet supported for hashtable-based caching\n";
            }
            else if (cache_type == CacheTrie) {
                out += "Storage key: Itemset\n";
                if (max_cache_size > NO_CACHE_LIMIT) {
                    out += "Cache limit: " + to_string(max_cache_size) + " --- wipe factor: " + custom_to_str(wipe_factor) + "\n";
                    switch (wipe_type) {
                        case Subnodes:
                            out += "Wipe criterion: Subnodes\n";
                            break;
                        case Recall:
                            out += "Wipe criterion: Recall\n";
                            break;
                        default:
                            out += "Wipe criterion: All nodes\n";
                    }
                }
            }
            
            nodeDataManager = new NodeDataManager_Trie(cover, tids_error_class_callback_pointer, supports_error_class_callback_pointer, tids_error_callback_pointer);
            searcher = new Search_trie_cache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, cache, maxError <= 0 ? NO_ERR : maxError, useSpecial, maxError <= 0 ? false : stopAfterError, similarlb, dynamic_branching, similar_for_branching, from_cpp);
            solution = new Solution_Trie(searcher);
        }
        if (from_cpp) { cout << out; out = ""; }
        searcher->run(); // perform the search
        Tree *tree_out = solution->getTree();
        tree_out->cacheSize = cache->getCacheSize();
        tree_out->runtime = duration<float>(high_resolution_clock::now() - startTime).count();
        out += tree_out->to_str();
//        cout << out << endl;
    }
    else {
        out += "Storage key: No cache";
        if (from_cpp) { cout << out; out = ""; }
        nodeDataManager = new NodeDataManager_Trie(cover, tids_error_class_callback_pointer, supports_error_class_callback_pointer, tids_error_callback_pointer);
        searcher = new Search_nocache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit,nullptr, maxError <= 0 ? NO_ERR : maxError, useSpecial,maxError > 0 && stopAfterError, use_ub);
        searcher->run(); // perform the search
        out += "runtime = " + to_string(duration<float>(high_resolution_clock::now() - startTime).count()) + "\n";
    }

    delete cache;
    delete nodeDataManager;
    delete cover;
    delete solution;
    delete searcher;
    delete dataReader;

    return out;
}
