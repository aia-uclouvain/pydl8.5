#ifndef QUERY_TOTALFREQ_H
#define QUERY_TOTALFREQ_H

#include <nodedataManager.h>
#include <vector>

struct Freq_NodeData {
    Attribute test;
    Freq_NodeData *left, *right;
    Error leafError;
    Error error;
    Error lowerBound;
    Size size;
    Depth solutionDepth;

    Freq_NodeData() {
        test = -1;
        left = nullptr;
        right = nullptr;
        leafError = FLT_MAX;
        error = FLT_MAX;
        lowerBound = 0;
        size = 1;
        solutionDepth = -1;
    }

    /*~QueryData_Best(){
        cout << "data is deleted" << endl;
        delete left;
        delete right;
    }*/
};

class Freq_NodeDataManager : public NodeDataManager {
public:
    Freq_NodeDataManager(
            RCover* cover,
                    function<vector<float>(RCover *)> *tids_error_class_callback = nullptr,
                    function<vector<float>(RCover *)> *supports_error_class_callback = nullptr,
                    function<float(RCover *)> *tids_error_callback = nullptr);

    ~Freq_NodeDataManager();

//    bool is_freq(pair<Supports, Support> supports);
//
//    bool is_pure(pair<Supports, Support> supports);

    bool updateData(NodeData *best, Error upperBound, Attribute attribute, NodeData *left, NodeData *right);

    NodeData *initData(Depth currentMaxDepth = -1, int hashcode = -1);

    LeafInfo computeLeafInfo();

    LeafInfo computeLeafInfo(Supports itemsetSupport);

    inline bool canimprove(NodeData *left, Error ub) { return ((Freq_NodeData *) left)->error < ub; }

    inline bool canSkip(NodeData *actualBest) { return floatEqual(((Freq_NodeData *) actualBest)->error, ((Freq_NodeData *) actualBest)->lowerBound); }

protected:
};

#endif