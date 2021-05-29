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

string search(Supports supports,
              Transaction ntransactions,
              Attribute nattributes,
              Class nclasses,
              Bool *data,
              Class *target,
              int maxdepth,
              int minsup,
              float maxError,
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
              int cache_size) {

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
    cout << "TrainingDistribution: ";
    forEachClass(i) cout << dataReader->getSupports()[i] << " ";
    cout << endl;

    if (save) return 0; // this param is not supported yet

    vector<float> weights;
    if (in_weights) weights = vector<float>(in_weights, in_weights + ntransactions);

    Cache *cache;
    switch (cache_type) {
        case CacheHash:
            cache = new Cache_Hash(nextPrime(cache_size), maxdepth);
            cout << "caching with hashmap limited to " << nextPrime(cache_size) << " elements" << endl;
            break;
        case CachePriority:
            cache = new Cache_Priority(nextPrime(cache_size), maxdepth);
            cout << "caching with priority queue limited to " << nextPrime(cache_size) << " elements" << endl;
            break;
        default:
            cache = new Cache_Trie(cache_size);
            ///cout << "caching with trie" << endl;
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

    string out = "";
    out = "(nItems, nTransactions) : ( " + to_string(dataReader->getNAttributes() * 2) + ", " +
          to_string(dataReader->getNTransactions()) + " )\n";


    void *lcm;
    if (iterative) {
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
    } else {
        lcm = new LcmPruned(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, cache, timeLimit,
                            continuousMap,
                            maxError <= 0 ? NO_ERR : maxError, maxError <= 0 ? false : stopAfterError);
        auto start_tree = high_resolution_clock::now();
        ((LcmPruned *) lcm)->run(); // perform the search
        auto stop_tree = high_resolution_clock::now();
        solution = new Freq_Solution(lcm, nodeDataManager);
        Tree *tree_out = solution->getTree();
        ((Freq_Tree *) tree_out)->cacheSize = ((LcmPruned *) lcm)->latticesize;
        ((Freq_Tree *) tree_out)->runtime = duration<double>(stop_tree - start_tree).count();
        out += ((Freq_Tree *) tree_out)->to_str();
//        out += "latsize : " + to_string(((Freq_Tree *) tree_out)->cacheSize) + "\n";
//        out += "maxdepth : " + to_string(maxdepth) + "\n";
//        out += "error : " + to_string(((Freq_NodeData *) ((LcmPruned *) lcm)->cache->root->data)->error) + "\n";
//        out += "runtime : " + to_string(duration<double>(stop_tree - start_tree).count()) + "\n";
//        out += "cachesize : " + to_string(cache->cachesize) + "\n";
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
