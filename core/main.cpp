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
    string datasetPath = "../datasets/dataset.txt";//"../../datasets/gaussian-mixture.txt";

    ifstream dataset(datasetPath);

    string line;
    int nfeatures = 5;

    std::cout << "Reading dataset..." << std::endl;
    // map<int, SupportClass> supports; // for each class, compute the number of transactions (support)
    vector<int> data; //data is a flatten 2D-array containing the values of features matrix while target is the array of target
    vector<double> target;

    // create an array of vectors, one for each attribute
    auto *data_tmp = new vector<int>[nfeatures];

    // read file from the second line and insert each value column by column in data_tmp
    // fill-in target array and supports map
    int counter = 0;
    while (getline(dataset, line)) {
        stringstream ss(line);
        string value;
        int i = 0;
        while (getline(ss, value, ',')) {
            if (i == nfeatures) {
                target.push_back(stod(value));
            } else {
                data_tmp[i].push_back(stoi(value));
            }
            i++;
        }

        counter++;
    }

    int ntransactions = counter;

    std::cout << "ntransactions: " << ntransactions << std::endl;

    // flatten the read data
    data.reserve(data_tmp[0].size() * nfeatures);
    for (int l = 0; l < nfeatures; ++l) {
        data.insert(data.end(), data_tmp[l].begin(), data_tmp[l].end());
    }
    delete[] data_tmp;

    int maxdepth = 3, minsup = 1;

    cout << "dataset: " << datasetPath.substr(datasetPath.find_last_of('/') + 1, datasetPath.find_last_of('.') - datasetPath.find_last_of('/') - 1) << endl;

    for (int i = 0; i <counter; i ++){
        cout << target[i] << " ";
    }

    constexpr double dropout = 0.9; // Chance of 0
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution dist(1 - dropout); // bernoulli_distribution takes chance of true n constructor

    vector<float> weight_vec(ntransactions);
    std::generate(weight_vec.begin(), weight_vec.end(), [&]{ return dist(gen); });


    string result;
        result = search(
                nullptr, //supports
                ntransactions, //ntransactions
                nfeatures, //nattributes
                0, //nclasses
                data.data(), //data
                nullptr, // classes
                target.data(), //float target
                3, //maxdepth
                1, //minsup
                nullptr, //maxError
                nullptr, //stopAfterError
                nullptr, //tids_error_class_callback
                nullptr, //supports_error_class_callback
                nullptr, //tids_error_callback
                nullptr, //in_weights
//                weight_vec.data(),
                true, //tids_error_class_is_null
                true, //supports_error_class_is_null
                true, //tids_error_is_null
                false, //infoGain
                false, //infoAsc
                false, //repeatSort
                QUANTILE_ERROR, // backup error
                new float[3]{0.2, 0.5, 0.8}, //quantiles
                3, //nquantiles
                0, //timeLimit
                true // verbose parameter
        );

    cout << result;

}