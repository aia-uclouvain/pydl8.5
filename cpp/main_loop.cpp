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
#include <filesystem>
#include <set>

using namespace std;
//namespace fs = __fs::filesystem;
namespace fs = filesystem;

float rand_0_1() {
    return (float) rand() / RAND_MAX;
}

float rand_a_b(int a, int b) {
    return (rand() % b) + a;
}

/*vector<float> generate_example_weights() {
    int n_instances = 3247;
    *//*vector<float>v;
    v.reserve(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        v.push_back(rand_0_1());
    }
    return v;*//*
    return vector<float>(n_instances, 1);
}
vector<float> get_training_error(string tree) {
    return vector<float>{30,5};
}*/

int main(int argc, char *argv[]) {
    srand(time(0));
    // string datasetname = argv[1]; int maxdepth = atoi(argv[2]);
    // string datasetPath = "../datasets/" + datasetname + ".txt";
    // cout << "file : " << datasetPath << " depth : " << maxdepth << endl;
    // string datasetPath = "../dl85_dist_source/datasets/tic-tac-toe.txt";
//     string datasetPath = "../dl85_dist_source/datasets/paper.txt";
//     string datasetPath = "../dl85_dist_source/datasets/paper_.txt";
//     string datasetPath = "../dl85_dist_source/datasets/paper_test.txt";
//    string datasetPath = "../dl85_dist_source/datasets/soybean.txt";
//    string datasetPath = "../../datasets/anneal.txt";
//    string datasetPath = "../dl85_dist_source/datasets/tic-tac-toe.txt";
//    string datasetPath = "../../datasets/tic-tac-toe__.txt";
//    string datasetPath = "../../datasets/soybean.txt";

    /*string datasets[] = {
            "anneal",
            "audiology",
            "australian-credit",
            "breast-wisconsin",
            "diabetes",
            "german-credit",
            "heart-cleveland",
            "hepatitis",
            "hypothyroid",
            "ionosphere",
            "kr-vs-kp",
            "letter",
            "lymph",
            "mushroom",
            "pendigits",
            "primary-tumor",
            "segment",
            "soybean",
            "splice-1",
            "tic-tac-toe",
            "vehicle",
            "vote",
            "yeast",
            "zoo-1"
    };*/

    struct rusage usage;
    float memory_amount = 0.f;

    string path = "../../datasets";
    set<fs::path> sortedFiles;
    for (const auto &entry : fs::directory_iterator(path))
        if (!entry.is_directory() && ((string) entry.path().c_str()).find(".DS_Store") == string::npos)
            sortedFiles.insert(entry.path());

    fstream csvStream;
    csvStream.open(path + "/tests/out.csv", ios::out | ios::app);
//    csvStream << "dataset,trie_without_time,hash_without_time,trie_without_memory,hash_without_memory,trie_with_time,hash_with_time,trie_with_memory,hash_with_memory\n";

//    for (const auto &datasetName : datasets) {
//        string datasetPath = "../../datasets/" + datasetName + ".txt";

    for (const auto &filename : sortedFiles) {
        string datasetPath = filename.c_str();
        string dataset_name = datasetPath.substr(datasetPath.find_last_of('/') + 1, datasetPath.find_last_of('.') - datasetPath.find_last_of('/') - 1);

//        if (dataset_name != "soybean") continue;
        if (dataset_name <= "german-credit") continue;

        csvStream << dataset_name;

        float infos[8];

        for (const auto &cache_size: {0, 1}) {

            for (const auto &cache_type: {CacheTrie, CacheHash}) {

                if (cache_type == CacheTrie) cout << "Cache Trie ";
                else cout << "Cache Hash ";

                if (cache_size == 0) cout << "without load" << endl;
                else cout << "with load" << endl;

                auto start_time = high_resolution_clock::now();

                /*for (const auto &datasetName : datasets) {
                    string datasetPath = "../../datasets/" + datasetName + ".txt";*/

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
                    data_tmp[k].push_back(
                            target[target.size() - 1]); // restore data saved in target array to its correct place
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

                auto *sup = new SupportClass[supports.size()];
                for (int j = 0; j < (int) supports.size(); ++j) sup[j] = supports[j];
                int ntransactions = (int) (data.size()) / nfeatures, nclass = (int) supports.size();
                int maxdepth = 3, minsup = 1, max_estimators = 1;
//    int cache_size = 50;
//    int cache_size = 10;
//    int cache_size = 3000000;
//        int cache_size = 0;
//    CacheType cache_type = CacheHash;
//        CacheType cache_type = CacheTrie;


                cout << "dataset: " << dataset_name << endl;
                /*function<vector<float>()> example_weights_callback = generate_example_weights;
                function<vector<float>(string)> predict_error_callback = get_training_error;*/
//                vector<float> in(3247, 1);
                string result;
                for (int i = 0; i < 1; ++i) {
                    result = search(
                            sup, //supports
                            ntransactions, //ntransactions
                            nfeatures, //nattributes
                            nclass, //nclasses
                            data.data(), //data
                            target.data(), //target
                            maxdepth, //maxdepth
                            minsup, //minsup
//                -1, //alpha
//                1, //gamma
                            0, //maxError
                            false, //stopAfterError
                            false, //iterative
                            nullptr, //tids_error_class_callback
                            nullptr, //supports_error_class_callback
                            nullptr, //tids_error_callback
                            nullptr, //in_weights
//                in.data(),
                            true, //tids_error_class_is_null
                            true, //supports_error_class_is_null
                            true, //tids_error_is_null
//                max_estimators,
                            true, //infoGain
                            true, //infoAsc
                            false, //repeatSort
                            0, //timeLimit
                            nullptr, //continuousMap
                            false, //save
//                 false, //uncomment on master branch — used for to activate caching lower bound
                            false,//, // verbose parameter
//                false
                            cache_type, //cache type
                            cache_size //cache size
                    );
                }

                cout << result;

                const auto get_used_memory = [&usage]() {
                    getrusage(RUSAGE_SELF, &usage);
                    return usage.ru_maxrss / 1024.f / 1024.f;
                };
                memory_amount = memory_amount == 0.f ? get_used_memory() : get_used_memory() - memory_amount;
                cout << "used memory: " << memory_amount << " Mb" << endl;

                cout << endl << endl;

                float timespan = duration<double>(high_resolution_clock::now() - start_time).count();
//                csvStream << "," << timespan;
                int ind = ((cache_type == CacheTrie) ? 0 : 1) + cache_size * 4;
                infos[ind] = timespan;
                infos[ind + 2] = memory_amount;

                /*istringstream stream1(result);
                string l;
                while (getline(stream1, l)){
                    if (l.find("RunTime") != string::npos) csvStream << dataset_name << "," << l.substr(l.find(": ") + 1) << "\n";
                    if (l.find("RunTime") != string::npos) csvStream << dataset_name << "," << l.substr(l.find(": ") + 1) << "\n";
                }*/




                /*struct rusage usage;
                getrusage(RUSAGE_SELF, &usage);
                cout << "used memory: " << usage.ru_maxrss / 1024.f / 1024.f << "Mb" << endl;*/

            }
        }

        for (const auto& val: infos) csvStream << "," << val;
        csvStream << "\n";
        csvStream.flush();
        if (datasetPath.find("hepatitis") != string::npos) break;
//        break;
    }
    csvStream.close();



    /*string path = "../../datasets";
    set<fs::path> sortedFiles;
    for (const auto & entry : fs::directory_iterator(path)) if (!entry.is_directory() && ((string)entry.path().c_str()).find(".DS_Store") == string::npos) sortedFiles.insert(entry.path());
    for (auto &filename : sortedFiles) cout << filename.c_str() << endl;*/


    /*std::ofstream myfile;
    myfile.open ("example.csv");
    myfile << "This is the first cell in the first column.\n";
    myfile << "a,b,c,\n";
    myfile << "c,s,v,\n";
    myfile << "1,2,3.456\n";
    myfile << "semi;colon";
    myfile.close();*/

}