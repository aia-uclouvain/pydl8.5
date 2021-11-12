#include "dl85.h"

using namespace std::chrono;

//bool verbose = false;

// Function that returns true if n is prime else returns false
bool isPrime(int n) {
    // Corner cases
    if (n <= 1) return false;
    if (n <= 3) return true;

    // This is checked so that we can skip middle five numbers in below loop
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;

    return true;
}

// Function to return the smallest prime number greater than N
int nextPrime(int N) {
    // Base case
    if (N <= 1) return 2;

    int prime = N;
    bool found = false;

    // Loop continuously until isPrime returns true for a number greater than n
    while (!found) {
        prime++;
        if (isPrime(prime)) found = true;
    }
    return prime;
}

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
              bool similar_for_branching) {

    auto start_time = high_resolution_clock::now(); // start the timer

    //as cython can't set null to function, we use a flag to set the apropriated functions to null in c++
    function<vector<float>(RCover *)> *tids_error_class_callback_pointer = &tids_error_class_callback;
    if (tids_error_class_is_null) tids_error_class_callback_pointer = nullptr;

    function<vector<float>(RCover *)> *supports_error_class_callback_pointer = &supports_error_class_callback;
    if (supports_error_class_is_null) supports_error_class_callback_pointer = nullptr;

    function<float(RCover *)> *tids_error_callback_pointer = &tids_error_callback;
    if (tids_error_is_null) tids_error_callback_pointer = nullptr;

    verbose = verbose_param;

    auto *dataReader = new DataManager(supports, ntransactions, nattributes, nclasses, data, target);
    cout << "nItems: " << nattributes*2 << " --- nTransactions: " << ntransactions << endl;
    cout << "Class Distribution: ";
    forEachClass(i) if (i == nclasses - 1) cout << i << ":" << dataReader->getSupports()[i]; else cout << i << ":" << dataReader->getSupports()[i] << ", ";
    cout << endl;

    vector<float> weights;
    if (in_weights) weights = vector<float>(in_weights, in_weights + ntransactions);

    Cache *cache;
    switch (cache_type) {
        case CacheHash:
//            cache = new Cache_Hash(nextPrime(cache_size), maxdepth);
            cache = new Cache_Hash(maxdepth, wipe_type, max_cache_size);
//            cout << "caching with hashmap limited to " << cache_size << " elements" << endl;
            break;
        case CachePriority:
            cache = new Cache_Priority(maxdepth, wipe_type, nextPrime(max_cache_size));
//            cout << "caching with priority queue limited to " << nextPrime(cache_size) << " elements" << endl;
            break;
        case CacheLtdTrie:
            cache = new Cache_Ltd_Trie(maxdepth, wipe_type, max_cache_size, wipe_factor);
            break;
        default:
            cache = new Cache_Trie(maxdepth, wipe_type, max_cache_size);
//            cout << "caching with trie" << endl;
    }
    // create an empty trie for the search space
//    Cache *cache = new Cache_Hash(1000003, maxdepth);
//    Cache *cache = new Cache_Hash(50021, maxdepth);
//    Cache *cache = new Cache_Trie();

    // use the correct cover depending on whether a weight array is provided or not
    RCover *cover;
    if (in_weights) cover = new RCoverWeight(dataReader, &weights);
    else cover = new RCoverFreq(dataReader); // non-weighted cover

    NodeDataManager *nodeDataManager = new NodeDataManagerFreq(cover, tids_error_class_callback_pointer,
                                                               supports_error_class_callback_pointer,
                                                               tids_error_callback_pointer);

    string out = "(nItems, nTransactions) : ( " + to_string(dataReader->getNAttributes() * 2) + ", " + to_string(dataReader->getNTransactions()) + " )\n";

    Search_base *searcher;
    Solution *solution = nullptr;
    if (with_cache) {
        searcher = new Search_cache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, cache, timeLimit, maxError <= 0 ? NO_ERR : maxError, useSpecial, maxError <= 0 ? false : stopAfterError, similarlb, dynamic_branching, similar_for_branching);
        searcher->run(); // perform the search
        solution = new SolutionFreq(searcher, nodeDataManager);
        Tree *tree_out = solution->getTree();
        ((Freq_Tree *) tree_out)->cacheSize = cache->getCacheSize();
        ((Freq_Tree *) tree_out)->runtime = duration<float>(high_resolution_clock::now() - start_time).count();
        out += ((Freq_Tree *) tree_out)->to_str();
    }
    else {
        searcher = new Search_nocache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, maxError <= 0 ? NO_ERR : maxError, useSpecial,maxError > 0 && stopAfterError, use_ub);
        searcher->run(); // perform the search
        out += "runtime = " + to_string(duration<float>(high_resolution_clock::now() - start_time).count()) + "\n";
    }

    delete cache;
    delete nodeDataManager;
    delete dataReader;
    delete cover;
    delete solution;

//    auto stop = high_resolution_clock::now();
//    cout << "Durée totale de l'algo : " << duration<double>(stop - start).count() << endl;

    return out;
}
