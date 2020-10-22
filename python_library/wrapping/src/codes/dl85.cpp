#include "dl85.h"

//using namespace std;
using namespace std::chrono;

bool verbose = false;


string search(Supports supports,
              Transaction ntransactions,
              Attribute nattributes,
              Class nclasses,
              Bool *data,
              Class *target,
              float maxError,
              bool stopAfterError,
              bool iterative,
              function<vector<float>(RCover *)> tids_error_class_callback,
              function<vector<float>(RCover *)> supports_error_class_callback,
              function<float(RCover *)> tids_error_callback,
              function<vector<float>()> example_weight_callback,
              function<vector<float>(string)> predict_error_callback,
              float* in_weights,
              bool tids_error_class_is_null,
              bool supports_error_class_is_null,
              bool tids_error_is_null,
              bool example_weight_is_null,
              bool predict_error_is_null,
//              bool in_weights_is_null,
              int maxdepth,
              int minsup,
              int max_estimators,
              bool infoGain,
              bool infoAsc,
              bool repeatSort,
              int timeLimit,
              map<int, pair<int, int>> *continuousMap,
              bool save,
              bool verbose_param) {

    auto start = high_resolution_clock::now(); // start the timer
    /*cout << "nattr = " << nattributes << endl;
    cout << "ntrans = " << ntransactions << endl;*/

    //as cython can't set null to function, we use a flag to set the apropriated functions to null in c++
    function<vector<float>(RCover *)> *tids_error_class_callback_pointer = &tids_error_class_callback;
    if (tids_error_class_is_null) tids_error_class_callback_pointer = nullptr;

    function<vector<float>(RCover *)> *supports_error_class_callback_pointer = &supports_error_class_callback;
    if (supports_error_class_is_null) supports_error_class_callback_pointer = nullptr;

    function<float(RCover *)> *tids_error_callback_pointer = &tids_error_callback;
    if (tids_error_is_null) tids_error_callback_pointer = nullptr;

    function<vector<float>()> *example_weight_callback_pointer = &example_weight_callback;
    if (example_weight_is_null) example_weight_callback_pointer = nullptr;

    function<vector<float>(string)> *predict_error_callback_pointer = &predict_error_callback;
    if (predict_error_is_null) predict_error_callback_pointer = nullptr;

    //cout << "print " << fast_error_callback->pyFunction << endl;
    verbose = verbose_param;
    string out = "";

    auto *dataReader = new DataManager(supports, ntransactions, nattributes, nclasses, data, target);

    if (save) return 0; // this param is not supported yet

    //create error object and initialize it in the next
    ExpError *experror;
    experror = new ExpError_Zero;

    /* set the relevant query for the search. If only one estimator is needed or
    weighter function is null, use total query. otherwise, use weighted query */

    Query *query = nullptr;
    Trie *trie = new Trie;
    RCover *cover = nullptr;
//    vector<float> weights(3247, 1);
    vector<float> weights;
    if (in_weights) weights = vector<float>(in_weights, in_weights + ntransactions);
    /*srand(time(0));
    int n_instances = 3247;
    weights.reserve(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        weights.push_back((float) rand()/RAND_MAX);
    }*/


    query = new Query_TotalFreq(trie,
                                          dataReader,
                                          experror,
                                          timeLimit,
                                          continuousMap,
                                          tids_error_class_callback_pointer,
                                          supports_error_class_callback_pointer,
                                          tids_error_callback_pointer,
                                          example_weight_callback_pointer,
                                          predict_error_callback_pointer,
                                          &weights,
                                          maxError,
                                          stopAfterError);


    query->maxdepth = maxdepth;
    query->minsup = minsup;

    out = "TrainingDistribution: ";
    forEachClass(i) out += std::to_string(dataReader->getSupports()[i]) + " ";
    out += "\n";
    out = "(nItems, nTransactions) : ( " + std::to_string(dataReader->getNAttributes() * 2) + ", " +
          std::to_string(dataReader->getNTransactions()) + " )\n";
    vector<Tree *> trees;
    float old_error_percentage = -1;

    void *lcm;
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
        trees.push_back(tree_out);
    }
    else {
        // the first run is always a totalfreq cover. It is used to speed up the error computation
        if (!in_weights) cover = new RCoverTotalFreq(dataReader);
        // we use weighted cover in case a weight array is passed as parameter
        else cover = new RCoverWeighted(dataReader);
        lcm = new LcmPruned(cover, query, infoGain, infoAsc, repeatSort);

        // perform the search
        auto start_tree = high_resolution_clock::now();
        ((LcmPruned *) lcm)->run();
        auto stop_tree = high_resolution_clock::now();

        // build the tree model
        Tree *tree_out = new Tree();
        query->printResult(tree_out);
        tree_out->latSize = ((LcmPruned *) lcm)->latticesize;
        tree_out->searchRt = duration<double>(stop_tree - start_tree).count();
        out += tree_out->to_str();

        float accuracy;
        float rho;
        vector<float>&& predict_return = vector<float>();
        if (predict_error_callback_pointer){
            predict_return = (*predict_error_callback_pointer)(tree_out->expression);
            accuracy = predict_return[0];
            rho = predict_return[1];
        }


        // print the tree
        // cout << tree_out->to_str() << endl;

        // add the tree to trees collection
        trees.push_back(tree_out);

        // set old_error_percentage in case of boosting
        old_error_percentage = -FLT_MAX;

        /*cout << "max_estim = " << max_estimators << endl;
        if (in_weights) cout << "in_weights not null" << in_weights << endl;
        else cout << "in_weights not null "  << endl;
        if (example_weight_callback_pointer) cout << "example pointer not null" << endl;
        else cout << "example pointer null" << endl;
        if (is_boosting) cout << "boosting actif" << endl;
        else cout << "boosting inactif" << endl;*/

        if (is_boosting) { //use the weighted cover for the next of searches because we are running boosting code

            for (int i = 1; i < max_estimators; ++i) {

                // set the delta value as big as possible to let the first iteration pass
                float delta_error_percentage = accuracy - old_error_percentage;
                /*if (delta_error_percentage < 5) break;*/

                vector<float>&& new_weights = (*example_weight_callback_pointer)();
                float gamma = new_weights.back();
                new_weights.pop_back();
                ((Query_TotalFreq*)query)->weights = &new_weights;
                // clear the trie
                delete trie; trie = new Trie; query->trie = trie;
                // create the weighted cover and set it to the search class
                cover = new RCoverWeighted(std::move(*((RCoverWeighted*)cover)));
                ((LcmPruned *) lcm)->cover = cover; ((LcmPruned *) lcm)->latticesize = 0;

                // perform the search
                start_tree = high_resolution_clock::now();
                ((LcmPruned *) lcm)->run();
                stop_tree = high_resolution_clock::now();

                // build the tree model object
                tree_out = new Tree();
                query->printResult(tree_out);
                tree_out->latSize = ((LcmPruned *) lcm)->latticesize;
                tree_out->searchRt = duration<double>(stop_tree - start_tree).count();
                out += tree_out->to_str();

                // print the tree
                // cout << tree_out->to_str() << endl;

                // add the tree to the list of trees
                trees.push_back(tree_out);

                // update old_error_percentage and current accuracy
                old_error_percentage = accuracy;
                predict_return = (*predict_error_callback_pointer)(tree_out->expression);
                accuracy = predict_return[0];
                rho = predict_return[1];
            }
        }
    }


    if (iterative) delete ((LcmIterative *) lcm);
    else delete ((LcmPruned *) lcm);
    delete trie;
    delete query;
    delete dataReader;
    delete cover;
    delete experror;
//    if (in_weights) delete [] in_weights;
    for (auto tree : trees) delete tree;

    auto stop = high_resolution_clock::now();
    cout << "Durée totale de l'algo : " << duration<double>(stop - start).count() << endl;

    return out;
}
