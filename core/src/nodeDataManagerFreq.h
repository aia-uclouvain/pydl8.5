#ifndef NODE_DATA_MANAGER_FREQ_H
#define NODE_DATA_MANAGER_FREQ_H

#include <nodeDataManager.h>
#include <vector>

struct Freq_NodeData : NodeData {

    Freq_NodeData(): NodeData() {}

};

class NodeDataManagerFreq : public NodeDataManager {
public:
    NodeDataManagerFreq(
            RCover* cover,
                    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
                    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
                    function<float(RCover *)> *tids_error_callback = nullptr);

    ~NodeDataManagerFreq();

//    bool is_freq(pair<Supports, Support> supports);
//
//    bool is_pure(pair<Supports, Support> supports);

    bool updateData(Node *best, Error upperBound, Attribute attribute, Node *left, Node *right, Cache* cache);

    NodeData *initData(RCover *cov = nullptr, Depth currentMaxDepth = -1, int hashcode = -1);

    LeafInfo computeLeafInfo(RCover *cov = nullptr);

    LeafInfo computeLeafInfo(ErrorVals itemsetSupport);

    inline bool canimprove(NodeData *left, Error ub) {
        return left->error < ub;
    }

    inline bool canSkip(NodeData *actualBest) { return floatEqual(actualBest->error, actualBest->lowerBound); }

protected:
};

#endif