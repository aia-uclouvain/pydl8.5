//
// Created by Gael Aglin on 2020-04-13.
//
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <iostream>
#include <functional>
#include "dl85.h"
#include "globals.h"

using namespace std;

typedef int Config;
#define optimized 0
#define basic 1

int getNFeatures(ifstream &dataset, vector<Class> &target, map<Class, ErrorVal> &supports_map) {
    // use the first line to count the number of features
    int nfeatures = -1, value;
    string line;
    getline(dataset, line); // read the first line of the file
    stringstream stream(line); // create a stream on the first line string
    while (stream >> value) {
        target.push_back(value); //use the target array to temporary store the values of the first line
        // increase the support per class based on the first line
        if (nfeatures == -1) {
            if (supports_map.find(value) == supports_map.end()) supports_map[value] = 1;
            else ++supports_map[value];
        }
        ++nfeatures;
    }
    return nfeatures;
}

void readFirstLine(vector<Bool> *data_per_feat, Attribute nfeatures, vector<Class> &target) {
    // read the first line stored in target value by getNFeatures function
    for (int k = nfeatures - 1; k >= 0; --k) {
        data_per_feat[k].push_back(target[target.size() - 1]); // restore data saved in target array to its correct place
        target.pop_back(); // each value copied is removed except for the last one which represents the target of the first line
    }
}

void readRemainingFileAndComputeSups(ifstream &dataset, vector<Class> &target, map<Class, ErrorVal> &supports_map,
                                     vector<Bool> *data_per_feat, Attribute nfeatures) {
    // read file from the second line and insert each value column by column in data_per_feat
    // fill-in target array and supports map
    int counter = 0, value;
    while (dataset >> value) {
        if (counter % (nfeatures + 1) == 0) { // first value on a new line
            target.push_back(value);
            if (supports_map.find(value) == supports_map.end()) supports_map[value] = 1;
            else ++supports_map[value];
        } else data_per_feat[(counter % (nfeatures + 1)) - 1].push_back(value);
        ++counter;
    }
}

vector<Bool> getFlattenedData(vector<Bool> *data_per_feat, int nfeatures) {
    vector<Bool> data;
    data.reserve(data_per_feat[0].size() * nfeatures);
    for (int l = 0; l < nfeatures; ++l) {
        data.insert(data.end(), data_per_feat[l].begin(), data_per_feat[l].end());
    }
    delete[] data_per_feat;
    return data;
}

ErrorVals getSupportPerClassArray(map<Class, ErrorVal> &supports_map) {
    auto *support_per_class = new ErrorVal[supports_map.size()];
    for (int j = 0; j < (int) supports_map.size(); ++j) support_per_class[j] = supports_map[j];
    return support_per_class;
}

int main(int argc, char *argv[]) {

    bool cli = true;
    //bool cli = false;
    string datasetPath;
    Config configuration;
    int maxdepth, minsup;

    if (cli){
        datasetPath = (argc > 1) ? std::string(argv[1]) : "../../datasets/anneal.txt";
//        datasetPath = (argc > 1) ? std::string(argv[1]) : "../../datasets/tests/bcb.txt";
        maxdepth = (argc > 2) ? std::stoi(argv[2]) : 4;
        configuration = (argc > 3 and std::string(argv[3]).find('b') == 0) ? basic : optimized;
        minsup = (argc > 4) ? std::stoi(argv[4]) : 1;
    }
    else {
        datasetPath = "../../datasets/anneal.txt";
        //configuration = basic;
        configuration = optimized;
        maxdepth = 5;
        minsup = 1;
    }

    CacheType cache_type;
    Size cache_size;
    WipeType wipe_type;
    float wipe_factor;
    bool with_cache, use_special_algo, verb, use_ub, sim_lb, dyn_branch, similar_for_branching;


    cache_type = CacheTrie;
    //cache_type = CacheHashCover;

    cache_size = NO_CACHE_LIMIT;
    //cache_size = 10000;

    wipe_type = Subnodes;
    //wipe_type = Recall;
    //wipe_type = All;

    wipe_factor = .4f;

    with_cache = true;
    //with_cache = false;

    //verb = true;
    verb = false;

    use_ub = true;
//    use_ub = false;



    switch (configuration) {
        case basic:
            use_special_algo = false;
            sim_lb = false;
            dyn_branch = false;
            similar_for_branching = false;
            break;
        case optimized:
            use_special_algo = true;
            sim_lb = true;
            dyn_branch = true;
            similar_for_branching = true;
            break;
        default:
            use_special_algo = true;
            sim_lb = true;
            dyn_branch = true;
            similar_for_branching = true;
    }

    ifstream dataset(datasetPath);
    map<Class, ErrorVal> supports_map; // for each class, compute the number of transactions (support)
    vector<Class> target; //data is a flatten 2D-array containing the values of features matrix while target is the array of target

    int nfeatures = getNFeatures(dataset, target, supports_map);
    auto *data_per_feat = new vector<Bool>[nfeatures]; // create an array of vectors, one for each attribute
    readFirstLine(data_per_feat, nfeatures, target);
    readRemainingFileAndComputeSups(dataset, target, supports_map, data_per_feat, nfeatures);
    auto ntransactions = (Transaction) (data_per_feat[0].size());
    auto nclass = (Class) supports_map.size();
    vector<Bool> data_flattened = getFlattenedData(data_per_feat, nfeatures);
    ErrorVals support_per_class = getSupportPerClassArray(supports_map);

    cout << "dataset: " << datasetPath.substr(datasetPath.find_last_of('/') + 1,datasetPath.find_last_of('.') - datasetPath.find_last_of('/') - 1) << endl;

    string result = launch(
            support_per_class, //supports
            ntransactions, //ntransactions
            nfeatures, //nattributes
            nclass, //nclasses
            data_flattened.data(), //data
            target.data(), //target
            maxdepth, //maxdepth
            minsup, //minsup
            0, //maxError
            false, //stopAfterError
            nullptr, //tids_error_class_callback
            nullptr, //supports_error_class_callback
            nullptr, //tids_error_callback
            nullptr, //sample_weight.data() in_weights
            true, //tids_error_class_is_null
            true, //supports_error_class_is_null
            true, //tids_error_is_null
            false, //infoGain
            false, //infoAsc
            false, //repeatSort
            0, //timeLimit
            verb, // verbose parameter
            cache_type, //cache type
            cache_size, //cache size
            wipe_type, // the type of wiping
            wipe_factor,
            with_cache,
            use_special_algo,
            use_ub,
            sim_lb,
            dyn_branch,
            similar_for_branching,
            true
    );

    deleteErrorVals(support_per_class);

    cout << result;
//    struct rusage usage{};
//    getrusage(RUSAGE_SELF, &usage);
//    cout << "used memory: " << usage.ru_maxrss / 1024.f / 1024.f << "Mb" << endl;

}