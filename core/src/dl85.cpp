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
              bool verbose_param) {

    //as cython can't set null to function, we use a flag to set the appropriated functions to null in c++
    function<vector<float>(RCover *)> *tids_error_class_callback_pointer = &tids_error_class_callback;
    if (tids_error_class_is_null) tids_error_class_callback_pointer = nullptr;

    function<vector<float>(RCover *)> *supports_error_class_callback_pointer = &supports_error_class_callback;
    if (supports_error_class_is_null) supports_error_class_callback_pointer = nullptr;

    function<float(RCover *)> *tids_error_callback_pointer = &tids_error_callback;
    if (tids_error_is_null) tids_error_callback_pointer = nullptr;

    verbose = verbose_param;
    string out = "";

    auto *dataReader = new DataManager(supports, ntransactions, nattributes, nclasses, data, target);


    vector<float> weights;
    if (in_weights) weights = vector<float>(in_weights, in_weights + ntransactions);

    // create an empty trie to store the search space
    Trie *trie = new Trie;

    Query *query = new Query_TotalFreq(minsup, maxdepth, trie, dataReader, timeLimit,
                                       tids_error_class_callback_pointer, supports_error_class_callback_pointer,
                                       tids_error_callback_pointer, maxError, stopAfterError);


    out = "(nFeats, nTransactions) : ( " + to_string(dataReader->getNAttributes()) + ", " + to_string(dataReader->getNTransactions()) + " )\n";
    out += "nInstances per class: ";
    forEachClass(i) {
        if (i == nclasses - 1) out += to_string(i) + ":" + custom_to_str(dataReader->getSupports()[i]);
        else out += to_string(i) + ":" + custom_to_str(dataReader->getSupports()[i]) + ", ";
    }
    // forEachClass(i) out += custom_to_str(dataReader->getSupports()[i]) + " ";
    out += "\n";

    // init variables
    // use the correct cover depending on whether a weight array is provided or not
    RCover *cover;
    if (in_weights) cover = new RCoverWeighted(dataReader, &weights); // weighted cover
    else cover = new RCoverTotalFreq(dataReader); // non-weighted cover
    auto lcm = new LcmPruned(cover, query, infoGain, infoAsc, repeatSort);
    auto start_tree = high_resolution_clock::now();
    ((LcmPruned *) lcm)->run(); // perform the search
    auto stop_tree = high_resolution_clock::now();
    Tree *tree_out = new Tree();
    query->printResult(tree_out); // build the tree model
    tree_out->latSize = ((LcmPruned *) lcm)->latticesize;
    tree_out->searchRt = duration<double>(stop_tree - start_tree).count();
    out += tree_out->to_str();


    delete trie;
    delete query;
    delete dataReader;
    delete cover;
    delete lcm;
    delete tree_out;

//    auto stop = high_resolution_clock::now();
//    cout << "Durée totale de l'algo : " << duration<double>(stop - start).count() << endl;

    return out;
}
