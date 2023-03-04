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
#include "argparse.cpp"

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

    argparse::ArgumentParser program("dl85");

//    program.add_argument("-h", "--datasetPath").help("The path of the dataset").default_value(string{"../../datasets/anneal.txt"});
//    program.add_argument("-h", "--datasetPath").help("The path of the dataset").default_value(string{"/Users/aglin/Downloads/iris_log_bin_2.txt"});
//    program.add_argument("-p", "--maxdepth").help("Maximum depth of the tree to learn").default_value(2).scan<'d', int>();
//    program.add_argument("-m", "--minsup").help("Minimum number of examples per leaf").default_value(1).scan<'d', int>();


    program.add_argument("datasetPath").help("The path of the dataset");
    program.add_argument("maxdepth").help("Maximum depth of the tree to learn").scan<'d', int>();
    program.add_argument("-m", "--minsup").help("Minimum number of examples per leaf").default_value(1).scan<'d', int>();
    program.add_argument("-x", "--maxerror").help("Initial upper bound. O to disable it.").default_value(0.f).scan<'f', float>();
    program.add_argument("-o", "--stopafterbetter").help("Stop search after finding better tree than maxerror").default_value(false).implicit_value(true);
    program.add_argument("-i", "--infogain").help("Use information to sort attributes order").default_value(false).implicit_value(true);
    program.add_argument("-g", "--infogainasc").help("Use ascendant order of information gain").default_value(false).implicit_value(true);
    program.add_argument("-r", "--repeatinfogainsort").help("Sort the attributes at each node").default_value(false).implicit_value(true);
    program.add_argument("-t", "--timelimit").help("Max runtime in seconds. O to disable it.").default_value(0).scan<'d', int>();
    program.add_argument("-c", "--cachetype").help("1- Trie + itemsets   2- Hashtable + itemsets   3- Hashtable + instances").default_value(1).scan<'d', int>();
    program.add_argument("-z", "--cachesize").help("The maximum size of the cache. O for unltd").default_value(0).scan<'d', int>();
    program.add_argument("-w", "--wipefactor").help("Cache percentage to free when it is full (between 0-1)").default_value(0.5f).scan<'f', float>();
    program.add_argument("-s", "--wipestrategy").help("1- Node reuses   2- Number of subnodes   3- All non-useful nodes").default_value(1).scan<'d', int>();
    program.add_argument("-n", "--nocache").help("Flag used to disable caching").default_value(false).implicit_value(true);
    program.add_argument("-v", "--verbose").help("Flag used to enable verbose").default_value(false).implicit_value(true);
    program.add_argument("-u", "--noub").help("Flag used to disable upper bound").default_value(false).implicit_value(true);
    program.add_argument("-a", "--nospecial").help("Flag used to disable specialized depth 2 algo").default_value(false).implicit_value(true);
    program.add_argument("-l", "--nosimilarity").help("Flag used to disable similarity lower bound").default_value(false).implicit_value(true);
    program.add_argument("-y", "--nodynamic").help("Flag used to disable dynamic branching").default_value(false).implicit_value(true);
    program.add_argument("-b", "--nosimbranching").help("Flag used to disable similarity lb for dynamic branching").default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto datasetPath = program.get<string>("datasetPath");
    CacheType cache_type;
    switch (program.get<int>("cachetype")) {
        case 3:
            cache_type = CacheHashCover;
            break;
        case 2:
            cache_type = CacheHashItemset;
            break;
        case 1:
            cache_type = CacheTrieItemset;
            break;
        default:
            cache_type = CacheTrieItemset;
    }

    WipeType wipe_type;
    switch (program.get<int>("wipestrategy")) {
        case 1:
            wipe_type = Reuses;
            break;
        case 2:
            wipe_type = Subnodes;
            break;
        case 3:
            wipe_type = All;
            break;
        default:
            wipe_type = Reuses;
    }

    ifstream dataset(datasetPath);
    if (not dataset) {
        cout << "The path you specified is not correct" << endl;
        exit(0);
    }

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

//    string dataname = datasetPath.substr(datasetPath.find_last_of('/') + 1,datasetPath.find_last_of('.') - datasetPath.find_last_of('/') - 1);
//    cout << "dataset: " << dataname << endl;

    string result = launch(
            support_per_class, //supports
            ntransactions, //ntransactions
            nfeatures, //nattributes
            nclass, //nclasses
            data_flattened.data(), //data
            target.data(), //target
            program.get<int>("maxdepth"), //maxdepth
            program.get<int>("minsup"), //minsup
            program.get<float>("maxerror"), //maxError
            program.get<bool>("stopafterbetter"), //stopAfterError
            nullptr, //tids_error_class_callback
            nullptr, //supports_error_class_callback
            nullptr, //tids_error_callback
            nullptr, //sample_weight.data() in_weights
            true, //tids_error_class_is_null
            true, //supports_error_class_is_null
            true, //tids_error_is_null
            program.get<bool>("infogain"), //infoGain
            program.get<bool>("infogainasc"), //infoAsc
            program.get<bool>("repeatinfogainsort"), //repeatSort
            program.get<int>("timelimit"), //timeLimit
            program.get<bool>("verbose"), // verbose parameter
            cache_type, //cache type
            program.get<int>("cachesize"), //cache size
            wipe_type, // the type of wiping
            program.get<float>("wipefactor"),
            !program.get<bool>("nocache"),
            !program.get<bool>("nospecial"),
            !program.get<bool>("noub"),
            !program.get<bool>("nosimilarity"),
            !program.get<bool>("nodynamic"),
            !program.get<bool>("nosimbranching"),
            true
    );

    deleteErrorVals(support_per_class);

    cout << result;

}