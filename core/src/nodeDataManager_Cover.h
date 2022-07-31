#ifndef NODE_DATA_MANAGER_COVER_FREQ_H
#define NODE_DATA_MANAGER_COVER_FREQ_H

#include "nodeDataManager.h"
#include "depthTwoNodeData.h"

struct HashCoverNode;

struct CoverNodeData : NodeData {

    HashCoverNode *left, *right;
    HashCoverNode *curr_left, *curr_right;

    CoverNodeData(): NodeData() {
        left = nullptr;
        right = nullptr;
        curr_left = nullptr;
        curr_right = nullptr;
    }

    CoverNodeData(const CoverNodeData& other): NodeData(other) {
        left = other.left;
        right = other.right;
        curr_left = other.curr_left;
        curr_right = other.curr_right;
    }

    CoverNodeData(const DepthTwo_NodeData& other) {
        test = other.test;
        leafError = other.leafError;
        error = other.error;
        lowerBound = other.lowerBound;
        size = other.size;
        left = (HashCoverNode*) other.left;
        right = (HashCoverNode*) other.right;
        curr_left = nullptr;
        curr_right = nullptr;
    }

    CoverNodeData& operator=(const CoverNodeData& other)
    {
        NodeData::operator=(other);
        left = other.left;
        right = other.right;
        curr_left = other.curr_left;
        curr_right = other.curr_right;
        return *this;
    }

    CoverNodeData& operator=(const DepthTwo_NodeData& other)
    {
        test = other.test;
        error = other.error;
        size = other.size;
        left = (HashCoverNode*) other.left;
        right = (HashCoverNode*) other.right;
        return *this;
    }

};

class NodeDataManager_Cover : public NodeDataManager {
public:
    explicit NodeDataManager_Cover(
            RCover* cover,
                    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
                    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
                    function<float(RCover *)> *tids_error_callback = nullptr);

    ~NodeDataManager_Cover() override;

    bool updateData(Node *best, Error upperBound, Attribute attribute, Node *left, Node *right, Itemset) override;

    NodeData *initData(RCover *cov = nullptr, Depth currentMaxDepth = -1, int hashcode = -1) override;

protected:
};

#endif