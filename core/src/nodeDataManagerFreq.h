#ifndef NODE_DATA_MANAGER_FREQ_H
#define NODE_DATA_MANAGER_FREQ_H

#include <nodeDataManager.h>
#include <vector>

struct Freq_NodeData {
    Attribute test;
    Freq_NodeData *left, *right;
    Error leafError;
    Error error;
    Error lowerBound;
    Size size;

    Freq_NodeData() {
        test = -1;
        left = nullptr;
        right = nullptr;
        leafError = FLT_MAX;
        error = FLT_MAX;
        lowerBound = 0;
        size = 1;
    }

    /*~QueryData_Best(){
        cout << "data is deleted" << endl;
        delete left;
        delete right;
    }*/
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
        return ((Freq_NodeData *) left)->error < ub;
    }

    inline bool canSkip(NodeData *actualBest) { return floatEqual(((Freq_NodeData *) actualBest)->error, ((Freq_NodeData *) actualBest)->lowerBound); }

protected:
};

#endif