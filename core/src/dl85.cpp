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

string launch(Supports supports,
              Transaction ntransactions,
              Attribute nattributes,
              Class nclasses,
              Bool *data,
              Class *target,
              Depth maxdepth,
              Support minsup,
              Error maxError,
              bool stopAfterError,
              bool iterative,
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
              map<int, pair<int, int>> *continuousMap,
              bool save,
              bool verbose_param,
              CacheType cache_type,
              int max_cache_size,
              WipeType wipe_type,
              float wipe_factor,
              bool with_cache,
              bool useSpecial,
              bool use_ub) {

//    auto start = high_resolution_clock::now(); // start the timer

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

    if (save) return 0; // this param is not supported yet

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
    if (in_weights) {
        cover = new RCoverWeighted(dataReader, &weights);
    } else {
        cover = new RCoverTotalFreq(dataReader); // non-weighted cover
    }

    NodeDataManager *nodeDataManager = new Freq_NodeDataManager(cover, tids_error_class_callback_pointer,
                                                                supports_error_class_callback_pointer,
                                                                tids_error_callback_pointer);

    Solution *solution = nullptr;

    string out = "(nItems, nTransactions) : ( " + to_string(dataReader->getNAttributes() * 2) + ", " +
          to_string(dataReader->getNTransactions()) + " )\n";


    Search_base *searcher;
    /*if (iterative) {
        lcm = new LcmIterative(dataReader, nodeDataManager, cache, infoGain, infoAsc, repeatSort, minsup, maxdepth,
                               timeLimit,
                               continuousMap, maxError <= 0 ? NO_ERR : maxError,
                               maxError <= 0 ? false : stopAfterError);
        auto start_tree = high_resolution_clock::now();
        ((LcmIterative *) lcm)->run();
        auto stop_tree = high_resolution_clock::now();
        solution = new Freq_Solution(lcm, nodeDataManager);
        Tree *tree_out = solution->getTree();
        ((Freq_Tree *) tree_out)->cacheSize = ((LcmIterative *) lcm)->latticesize;
        ((Freq_Tree *) tree_out)->runtime = duration<double>(stop_tree - start_tree).count();
        out += ((Freq_Tree *) tree_out)->to_str();
    } else {*/

    if (with_cache) {
        searcher = new Search_cache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, cache, timeLimit, continuousMap, maxError <= 0 ? NO_ERR : maxError, useSpecial, maxError <= 0 ? false : stopAfterError);
        auto start_tree = high_resolution_clock::now();
        searcher->run(); // perform the search
        auto stop_tree = high_resolution_clock::now();
        solution = new Freq_Solution(searcher, nodeDataManager);
        Tree *tree_out = solution->getTree();
//        ((Freq_Tree *) tree_out)->cacheSize = ((LcmPruned *) lcm)->latticesize;
        ((Freq_Tree *) tree_out)->cacheSize = cache->getCacheSize();
        ((Freq_Tree *) tree_out)->runtime = duration<float>(stop_tree - start_tree).count();
        out += ((Freq_Tree *) tree_out)->to_str();
//        out += "latsize : " + to_string(((Freq_Tree *) tree_out)->cacheSize) + "\n";
//        out += "maxdepth : " + to_string(maxdepth) + "\n";
//        out += "error : " + to_string(((Freq_NodeData *) ((LcmPruned *) lcm)->cache->root->data)->error) + "\n";
//        out += "runtime : " + to_string(duration<double>(stop_tree - start_tree).count()) + "\n";
//        out += "cachesize : " + to_string(cache->cachesize) + "\n";
    }
    else {
        searcher = new Search_nocache(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, maxError <= 0 ? NO_ERR : maxError, useSpecial,maxError > 0 && stopAfterError, use_ub);
        auto start_tree = high_resolution_clock::now();
        searcher->run(); // perform the search
        auto stop_tree = high_resolution_clock::now();
        out += "runtime = " + to_string(duration<float>(stop_tree - start_tree).count()) + "\n";
    }

//    }

    delete cache;
    delete nodeDataManager;
    delete dataReader;
    delete cover;
    delete solution;

//    auto stop = high_resolution_clock::now();
//    cout << "Durée totale de l'algo : " << duration<double>(stop - start).count() << endl;

    return out;
}
