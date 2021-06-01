#include "dl85.h"

using namespace std::chrono;

//bool verbose = false;

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
              float* in_weights,
              bool tids_error_class_is_null,
              bool supports_error_class_is_null,
              bool tids_error_is_null,
              bool infoGain,
              bool infoAsc,
              bool repeatSort,
              int timeLimit,
              map<int, pair<int, int>> *continuousMap,
              bool save,
              bool verbose_param) {

//    auto start = high_resolution_clock::now(); // start the timer

    //as cython can't set null to function, we use a flag to set the apropriated functions to null in c++
    function<vector<float>(RCover *)> *tids_error_class_callback_pointer = &tids_error_class_callback;
    if (tids_error_class_is_null) tids_error_class_callback_pointer = nullptr;

    function<vector<float>(RCover *)> *supports_error_class_callback_pointer = &supports_error_class_callback;
    if (supports_error_class_is_null) supports_error_class_callback_pointer = nullptr;

    function<float(RCover *)> *tids_error_callback_pointer = &tids_error_callback;
    if (tids_error_is_null) tids_error_callback_pointer = nullptr;

    verbose = verbose_param;
    string out = "";

    auto *dataReader = new DataManager(supports, ntransactions, nattributes, nclasses, data, target);

    if (save) return 0; // this param is not supported yet

    //create error object and initialize it in the next
    ExpError *experror;
    experror = new ExpError_Zero;

    vector<float> weights;
    if (in_weights) weights = vector<float>(in_weights, in_weights + ntransactions);

    // create an empty trie for the search space
    Trie *trie = new Trie;

    Query *query = new Query_TotalFreq(minsup, maxdepth, trie, dataReader, experror, timeLimit, continuousMap,
                                       tids_error_class_callback_pointer, supports_error_class_callback_pointer,
                                       tids_error_callback_pointer, maxError, stopAfterError);

    out = "TrainingDistribution: ";
    forEachClass(i) out += std::to_string(dataReader->getSupports()[i]) + " ";
    out += "\n";
    out = "(nItems, nTransactions) : ( " + to_string(dataReader->getNAttributes() * 2) + ", " + to_string(dataReader->getNTransactions()) + " )\n";

    // init variables
    RCover *cover = nullptr; void *lcm;

    if (iterative) {
        lcm = new LcmIterative(dataReader, query, trie, infoGain, infoAsc, repeatSort);
        auto start_tree = high_resolution_clock::now();
        ((LcmIterative *) lcm)->run();
        auto stop_tree = high_resolution_clock::now();
        Tree *tree_out = new Tree();
        query->printResult(tree_out);
        tree_out->latSize = ((LcmIterative *) lcm)->latticesize;
        tree_out->searchRt = duration<double>(stop_tree - start_tree).count();
        out += tree_out->to_str();
    }
    else {
        // use the correct cover depending on whether a weight array is provided or not
        if (in_weights) cover = new RCoverWeighted(dataReader, &weights); // weighted cover
        else cover = new RCoverTotalFreq(dataReader); // non-weighted cover
        lcm = new LcmPruned(cover, query, infoGain, infoAsc, repeatSort);
        auto start_tree = high_resolution_clock::now();
        ((LcmPruned *) lcm)->run(); // perform the search
        auto stop_tree = high_resolution_clock::now();
        Tree *tree_out = new Tree();
        query->printResult(tree_out); // build the tree model
        tree_out->latSize = ((LcmPruned *) lcm)->latticesize;
        tree_out->searchRt = duration<double>(stop_tree - start_tree).count();
        out += tree_out->to_str();
        if (query->timeLimitReached) out += "Timeout: True\n";
        else out += "Timeout: False\n";
    }

    delete trie;
    delete query;
    delete dataReader;
    delete cover;
    delete experror;

//    auto stop = high_resolution_clock::now();
//    cout << "Durée totale de l'algo : " << duration<double>(stop - start).count() << endl;

    return out;
}
