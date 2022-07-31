#ifndef NODE_DATA_MANAGER_TRIE_FREQ_H
#define NODE_DATA_MANAGER_TRIE_FREQ_H

#include "nodeDataManager.h"
#include "depthTwoNodeData.h"
//#include "node_trie.h"
//#include <vector>

//struct DepthTwo_NodeData;

struct TrieNodeData : NodeData {
    Attribute curr_test;
    TrieNodeData(): NodeData() {
        curr_test = -1;
    }

    TrieNodeData(const TrieNodeData& other): NodeData(other) {
        curr_test = other.curr_test;
    }

    TrieNodeData(const DepthTwo_NodeData& other) {
        test = other.test;
        leafError = other.leafError;
        error = other.error;
        lowerBound = other.lowerBound;
        size = other.size;
        curr_test = -1;
    }

    TrieNodeData& operator=(const TrieNodeData& other) {
        NodeData::operator=(other);
        curr_test = other.curr_test;
        return *this;
    }

    TrieNodeData& operator=(const DepthTwo_NodeData& other) {
        test = other.test;
        error = other.error;
        size = other.size;
        return *this;
    }

};

class NodeDataManager_Trie : public NodeDataManager {
public:
    explicit NodeDataManager_Trie(
            RCover* cover,
                    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
                    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
                    function<float(RCover *)> *tids_error_callback = nullptr);

    ~NodeDataManager_Trie() override;

    bool updateData(Node *best, Error upperBound, Attribute attribute, Node *left, Node *right, Itemset) override;

    NodeData *initData(RCover *cov = nullptr, Depth currentMaxDepth = -1, int hashcode = -1) override;

protected:
};

#endif