#ifndef NODE_DATA_MANAGER_H
#define NODE_DATA_MANAGER_H

#include <utility>
#include "globals.h"
#include "rCover.h"
#include "dataManager.h"
#include <iostream>
#include <cfloat>
#include <functional>
#include <vector>
#include <chrono>

struct Node;

class Cache;

using namespace std;
using namespace std::chrono;

//typedef void *NodeData; // using void pointers is much lighter than class derivation

struct NodeData {
    Attribute test;
    Attribute curr_test;
    Node *left, *right;
    Error leafError;
    Error error;
    Error lowerBound;
    Size size;

    NodeData() {
        test = INT32_MAX;
        curr_test = -1;
        left = nullptr;
        right = nullptr;
        leafError = FLT_MAX;
        error = FLT_MAX;
        lowerBound = 0;
        size = 1;
    }

    NodeData(const NodeData& other) {
        test = other.test;
        curr_test = other.curr_test;
        left = other.left;
        right = other.right;
        leafError = other.leafError;
        error = other.error;
        lowerBound = other.lowerBound;
        size = other.size;
    }

    NodeData& operator=(const NodeData& other)
    {
        test = other.test;
        left = other.left;
        right = other.right;
        error = other.error;
        size = other.size;
        return *this;
    }

    virtual ~NodeData() {}
};

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

class NodeDataManager {
public:
    NodeDataManager(RCover* cover,
          function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
          function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
          function<float(RCover *)> *tids_error_callback = nullptr);

    virtual ~NodeDataManager();

//    virtual bool is_freq(pair<Supports, Support> supports) = 0;
//
//    virtual bool is_pure(pair<Supports, Support> supports) = 0;

    virtual bool canimprove(NodeData *left, Error ub) = 0;

    virtual bool canSkip(NodeData *actualBest) = 0;

    virtual NodeData *initData(RCover *cov = nullptr, Depth currentMaxDepth = -1, int hashcode = -1) = 0;

    virtual LeafInfo computeLeafInfo(RCover *cov = nullptr) = 0;

    virtual LeafInfo computeLeafInfo(ErrorVals itemsetSupport) = 0;

//    virtual bool updateData(Node *best, Error upperBound, Attribute attribute, Node *left, Node *right, Cache* cache = nullptr) = 0;
    virtual bool updateData(Node *best, Error upperBound, Attribute attribute, Node *left, Node *right, Cache* cache = nullptr, Itemset = Itemset()) = 0;

//    virtual void printResult(Tree *tree) = 0;

//    void setStartTime() { startTime = high_resolution_clock::now(); }


    RCover* cover;
    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr;
    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr;
    function<float(RCover *)> *tids_error_callback = nullptr;

};

#endif