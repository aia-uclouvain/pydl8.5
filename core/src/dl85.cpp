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

    GlobalParams::getInstance()->verbose = verbose_param;
//    if (verbose_param) Logger::setTrue();
//    else Logger::setFalse();

    auto *dataReader = new DataManager(supports, ntransactions, nattributes, nclasses, data, target);

    GlobalParams::getInstance()->out = "(nFeats, nTransactions) : ( " + to_string(dataReader->getNAttributes()) + ", " + to_string(dataReader->getNTransactions()) + " )\n";
    GlobalParams::getInstance()->out += "Class Distribution: ";
    forEachClass(i) {
        if (i == nclasses - 1) GlobalParams::getInstance()->out += to_string(i) + ":" + custom_to_str(dataReader->getSupports()[i]);
        else GlobalParams::getInstance()->out += to_string(i) + ":" + custom_to_str(dataReader->getSupports()[i]) + ", ";
    }
    GlobalParams::getInstance()->out += "\nmaxdepth: " + to_string(maxdepth) + " --- minsup: " + to_string(minsup) + "\n";

    vector<float> weights;
    if (in_weights) weights = vector<float>(in_weights, in_weights + ntransactions);

    // use the correct cover depending on whether a weight array is provided or not
    RCover *cover;
    if (in_weights) cover = new RCoverWeight(dataReader, &weights);
    else cover = new RCoverFreq(dataReader); // non-weighted cover

    Search_base *searcher;
    Solution *solution;
    NodeDataManager *nodeDataManager;

    if (with_cache) {
        Cache *cache;

        if (cache_type == CacheHashCover) {
            GlobalParams::getInstance()->out += "Cache type: Hashtable\n";
            GlobalParams::getInstance()->out += "Cache key: Covers\n";
            if (max_cache_size > NO_CACHE_LIMIT) {
                GlobalParams::getInstance()->out += "Cache boundary is not yet supported for cache based on hashtable and/or using cover-based keys\n";
                max_cache_size = NO_CACHE_LIMIT;
            }
            cache = new Cache_Hash_Cover(maxdepth, wipe_type, max_cache_size, wipe_factor);
            nodeDataManager = new NodeDataManager_Cover(cover, tids_error_class_callback_pointer, supports_error_class_callback_pointer, tids_error_callback_pointer);
            searcher = new Search_cover_cache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, cache, maxError <= 0 ? NO_ERR : maxError, useSpecial, maxError <= 0 ? false : stopAfterError, similarlb, dynamic_branching, similar_for_branching, from_cpp);
            solution = new Solution_Cover(searcher);
        }
        else {
            if (cache_type == CacheHashItemset) {
                GlobalParams::getInstance()->out += "Cache type: Hashtable\n";
                GlobalParams::getInstance()->out += "Cache key: Frequent Itemsets\n";
                if (max_cache_size > NO_CACHE_LIMIT) {
                    GlobalParams::getInstance()->out += "Cache boundary is not yet supported for cache based on hashtable and/or using cover-based keys\n";
                    max_cache_size = NO_CACHE_LIMIT;
                }
                cache = new Cache_Hash_Itemset(maxdepth, wipe_type, max_cache_size, wipe_factor);
            }
            else { // if (cache_type == CacheTrieItemset)
                GlobalParams::getInstance()->out += "Cache type: Trie\n";
                GlobalParams::getInstance()->out += "Cache key: Frequent Itemsets\n";
                if (max_cache_size > NO_CACHE_LIMIT) {
                    GlobalParams::getInstance()->out += "Cache size limit: " + to_string(max_cache_size) + " --- wipe factor: " + custom_to_str(wipe_factor) + "\n";
                    switch (wipe_type) {
                        case Subnodes:
                            GlobalParams::getInstance()->out += "Wipe criterion: Subnodes\n";
                            break;
                        case Reuses:
                            GlobalParams::getInstance()->out += "Wipe criterion: Recall\n";
                            break;
                        default:
                            GlobalParams::getInstance()->out += "Wipe criterion: All nodes\n";
                    }
                }
                cache = new Cache_Trie(maxdepth, wipe_type, max_cache_size, wipe_factor);
            }
            
            nodeDataManager = new NodeDataManager_Trie(cover, tids_error_class_callback_pointer, supports_error_class_callback_pointer, tids_error_callback_pointer);
            searcher = new Search_trie_cache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, cache, maxError <= 0 ? NO_ERR : maxError, useSpecial, maxError <= 0 ? false : stopAfterError, similarlb, dynamic_branching, similar_for_branching, from_cpp);
            solution = new Solution_Trie(searcher);
        }
        if (from_cpp) { cout << GlobalParams::getInstance()->out; GlobalParams::getInstance()->out = ""; }
        searcher->run(); // perform the search
        Tree *tree_out = solution->getTree();
        tree_out->cacheSize = cache->getCacheSize();
        tree_out->runtime = duration<float>(high_resolution_clock::now() - GlobalParams::getInstance()->startTime).count();
        GlobalParams::getInstance()->out += tree_out->to_str();

        delete cache;
    }
    else {
        GlobalParams::getInstance()->out += "Storage key: No cache";
        if (from_cpp) { cout << GlobalParams::getInstance()->out; GlobalParams::getInstance()->out = ""; }
        nodeDataManager = new NodeDataManager_Trie(cover, tids_error_class_callback_pointer, supports_error_class_callback_pointer, tids_error_callback_pointer);
        searcher = new Search_nocache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit,nullptr, maxError <= 0 ? NO_ERR : maxError, useSpecial,maxError > 0 && stopAfterError, use_ub);
        searcher->run(); // perform the search
        GlobalParams::getInstance()->out += "runtime = " + to_string(duration<float>(high_resolution_clock::now() - GlobalParams::getInstance()->startTime).count()) + "\n";
    }


    delete nodeDataManager;
    delete cover;
    if (with_cache) delete solution;
    delete searcher;
    delete dataReader;

    string out_cpy = GlobalParams::getInstance()->out;
    GlobalParams::free();

    return out_cpy;
}
