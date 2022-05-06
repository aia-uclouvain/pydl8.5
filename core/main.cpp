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
#include <random>

using namespace std;

int main(int argc, char *argv[]) {
    string datasetPath = "../../datasets/anneal.txt";

    ifstream dataset(datasetPath);
    string line;
    int nfeatures = -1, value;
    map<int, SupportClass> supports; // for each class, compute the number of transactions (support)
    vector<int> data, target; //data is a flatten 2D-array containing the values of features matrix while target is the array of target

    // read the number of features
    getline(dataset, line); // read the first line of the file
    stringstream stream(line); // create a stream on the first line string
    while (stream >> value) {
        target.push_back(value); //use temporary the target array to store the values of the first line
        if (nfeatures == -1) {
            if (supports.find(value) == supports.end()) supports[value] = 1;
            else ++supports[value];
        }
        ++nfeatures;
    }

    // create an array of vectors, one for each attribute
    auto *data_tmp = new vector<int>[nfeatures];
    for (int k = nfeatures - 1; k >= 0; --k) {
        data_tmp[k].push_back(target[target.size() - 1]); // restore data saved in target array to its correct place
        target.pop_back(); // each value copied is removed except for the last one which represents the target of the first line
    }

    // read file from the second line and insert each value column by column in data_tmp
    // fill-in target array and supports map
    int counter = 0;
    while (dataset >> value) {
        if (counter % (nfeatures + 1) == 0) { // first value on a new line
            target.push_back(value);
            if (supports.find(value) == supports.end()) supports[value] = 1;
            else ++supports[value];
        } else data_tmp[(counter % (nfeatures + 1)) - 1].push_back(value);
        ++counter;
    }

    // flatten the read data
    data.reserve(data_tmp[0].size() * nfeatures);
    for (int l = 0; l < nfeatures; ++l) {
        data.insert(data.end(), data_tmp[l].begin(), data_tmp[l].end());
    }
    delete[] data_tmp;

    auto *sup = new SupportClass [supports.size()];
    for (int j = 0; j < (int) supports.size(); ++j) sup[j] = supports[j];
    int ntransactions = (int) (data.size()) / nfeatures, nclass = (int) supports.size();
    int maxdepth = 3, minsup = 1;

    cout << "dataset: " << datasetPath.substr(datasetPath.find_last_of('/') + 1, datasetPath.find_last_of('.') - datasetPath.find_last_of('/') - 1) << endl;

    string result;
        result = search(
                sup, //supports
                ntransactions, //ntransactions
                nfeatures, //nattributes
                nclass, //nclasses
                data.data(), //data
                target.data(), //target
                maxdepth, //maxdepth
                minsup, //minsup
                0, //maxError
                false, //stopAfterError
                nullptr, //tids_error_class_callback
                nullptr, //supports_error_class_callback
                nullptr, //tids_error_callback
                nullptr, //in_weights
                true, //tids_error_class_is_null
                true, //supports_error_class_is_null
                true, //tids_error_is_null
                false, //infoGain
                false, //infoAsc
                false, //repeatSort
                0, //timeLimit
                false // verbose parameter
        );


    delete[] sup;
    cout << result;

}