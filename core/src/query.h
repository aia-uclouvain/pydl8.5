#ifndef QUERY_H
#define QUERY_H

#include <utility>
#include "globals.h"
#include "rCover.h"
#include "dataManager.h"
#include <iostream>
#include <cfloat>
#include <functional>
#include <vector>
#include <chrono>

struct TrieNode;

class Trie;

using namespace std;
using namespace std::chrono;

typedef void *QueryData; // using void pointers is much lighter than class derivation

/**
 * ErrorValues - this structure represents the important values computed at a leaf node; mainly the error and the class
 * @param error - the error computed at the leaf node
 * @param lowerb - the lowerbound of the error at current code. It will be removed since the error will be computed only a leaf node
 * @param conflict - it is set to 1 to express that several classes have the same maximum support, otherwise it is worth 0
 * @param corrects - special array of support per class; the non-majority classes supports are set to 0
 * @param falses - special array of support per class; the majority class support is set to 0
 */
struct LeafInfo {
    Error error;
    Class maxclass;
};

/**
 * This structure a decision tree model learnt from input data
 * @param expression - a json string representing the tree
 * @param size - the number of nodes (branches + leaves) in the tree
 * @param depth - the depth of the tree; the length of the longest rule in the tree
 * @param trainingError - the error of the tree on the training set given the used objective function
 * @param latSize - the number of nodes explored before finding the solution. Currently this value is not correct :-(
 * @param searchRt - the time that the search took
 * @param timeout - a boolean variable to represent the fact that the search reached a timeout or not
 */
struct Tree {
    string expression;
    int size;
    Depth depth;
    Error trainingError;
    int latSize;
    float searchRt;
    float accuracy;
    bool timeout;


    string to_str() const {
        string out = "";
        out += "Tree: " + expression + "\n";
        if (expression != "(No such tree)") {
            out += "Size: " + to_string(size) + "\n";
            out += "Depth: " + to_string(depth) + "\n";
            out += "Error: " + custom_to_str(trainingError) + "\n";
            out += "Accuracy: " + custom_to_str(accuracy) + "\n";
        }
        out += "LatticeSize: " + to_string(latSize) + "\n";
        out += "RunTime: " + custom_to_str(searchRt) + "\n";
        if (timeout) out += "Timeout: True\n";
        else out += "Timeout: False\n";
        return out;
    }
};

class Query {
public:
    Query(Support minsup,
          Depth maxdepth,
          Trie *trie,
          DataManager *dm,
          int timeLimit,
          function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
          function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
          function<float(RCover *)> *tids_error_callback = nullptr,
          float maxError = NO_ERR,
          bool stopAfterError = false);

    virtual ~Query();

    virtual bool is_freq(pair<Supports, Support> supports) = 0;

    virtual bool is_pure(pair<Supports, Support> supports) = 0;

    virtual bool canimprove(QueryData *left, Error ub) = 0;

    virtual bool canSkip(QueryData *actualBest) = 0;

    virtual QueryData *initData(RCover *tid, Depth currentMaxDepth = -1) = 0;

    virtual LeafInfo computeLeafInfo(RCover *cover) = 0;

    virtual LeafInfo computeLeafInfo(Supports itemsetSupport) = 0;

    virtual bool updateData(QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right) = 0;

    virtual void printResult(Tree *tree) = 0;

    void setStartTime() { startTime = high_resolution_clock::now(); }

    DataManager *dm; // we need to have information about the data for default predictions
    Trie *trie;
    TrieNode *realroot; // as the empty itemset may not have an empty closure
    Support minsup;
    Depth maxdepth;
    time_point<high_resolution_clock> startTime;
    int timeLimit;
    bool timeLimitReached = false;
    float maxError = NO_ERR;
    bool stopAfterError = false;
    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr;
    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr;
    function<float(RCover *)> *tids_error_callback = nullptr;

};

#endif