//
// Created by Gael Aglin on 2020-04-13.
//
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <iostream>
#include <functional>
#include <ctime>
#include <cstdlib>
#include "dl85.h"
#include "globals.h"
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;

int getNFeatures(ifstream &dataset, vector<Class> &target, map<Class, SupportClass> &supports_map) {
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

void readRemainingFileAndComputeSups(ifstream &dataset, vector<Class> &target, map<Class, SupportClass> &supports_map,
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

Supports getSupportPerClassArray(map<Class, SupportClass> &supports_map) {
    auto *support_per_class = new SupportClass[supports_map.size()];
    for (int j = 0; j < (int) supports_map.size(); ++j) support_per_class[j] = supports_map[j];
    return support_per_class;
}

int functorExample(vector<float>& vec) {
    return vec.size();
}

int main(int argc, char *argv[]) {

    bool cli = false;
    string datasetPath;
    int maxdepth, minsup;

    if (cli){
        datasetPath = (argc > 1) ? "../datasets/" + std::string(argv[1]) + ".txt" : "../datasets/anneal.txt";
        maxdepth = (argc > 2) ? atoi(argv[2]) : 2;
        minsup = (argc > 3) ? atoi(argv[3]) : 1;
    }
    else {
        //datasetPath = "../datasets/tic-tac-toe.txt";
        datasetPath = "../../datasets/soybean.txt";
//        datasetPath = "../datasets/hepatitis.txt";
//        datasetPath = "../datasets/tests/paper.txt";
//        datasetPath = "../datasets/tic-tac-toe.txt";
        maxdepth = 4;
        minsup = 1;
    }

//    CacheType cache_type = CacheTrie;
    CacheType cache_type = CacheLtdTrie;
    Size cache_size = 500000;
    WipeType wipe_type = WipeAll;
//    Size cache_size = NO_CACHE_LIMIT;
//    int cache_size = 10;
//    int cache_size = 3000000;
//    CacheType cache_type = CacheHash;

    ifstream dataset(datasetPath);
    map<Class, SupportClass> supports_map; // for each class, compute the number of transactions (support)
    vector<Class> target; //data is a flatten 2D-array containing the values of features matrix while target is the array of target

    int nfeatures = getNFeatures(dataset, target, supports_map);
    auto *data_per_feat = new vector<Bool>[nfeatures]; // create an array of vectors, one for each attribute
    readFirstLine(data_per_feat, nfeatures, target);
    readRemainingFileAndComputeSups(dataset, target, supports_map, data_per_feat, nfeatures);
    auto ntransactions = (Transaction) (data_per_feat[0].size());
    auto nclass = (Class) supports_map.size();
    vector<Bool> data_flattened = getFlattenedData(data_per_feat, nfeatures);
    Supports support_per_class = getSupportPerClassArray(supports_map);

    cout << "dataset: " << datasetPath.substr(datasetPath.find_last_of('/') + 1,datasetPath.find_last_of('.') - datasetPath.find_last_of('/') - 1) << endl;

    /*function<vector<float>()> example_weights_callback = generate_example_weights;
    function<vector<float>(string)> predict_error_callback = get_training_error;*/
    //function<int(vector<float>&)> callback = functorExample; //params type are in brackets while return type come before
    //vector<float> sample_weight(ntransactions, 1);

    string result = search(
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
            false, //iterative
            nullptr, //tids_error_class_callback
            nullptr, //supports_error_class_callback
            nullptr, //tids_error_callback
            nullptr, //sample_weight.data() in_weights
            true, //tids_error_class_is_null
            true, //supports_error_class_is_null
            true, //tids_error_is_null
            true, //infoGain
            true, //infoAsc
            false, //repeatSort
            0, //timeLimit
            nullptr, //continuousMap
            false, //save
            false, // verbose parameter
            cache_type, //cache type
            cache_size, //cache size
            wipe_type // the type of wiping
    );

    cout << result;
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "used memory: " << usage.ru_maxrss / 1024.f / 1024.f << "Mb" << endl;

}