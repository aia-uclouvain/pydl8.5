#ifndef SEARCH_COVER_CACHE_H
#define SEARCH_COVER_CACHE_H

#include "search_base.h"
#include "nodeDataManager_Cover.h"

typedef bool HasInter;

struct SimilarValss{
    bitset<M>* s_cover;
    Support s_coversize;
    Error s_error;
    int* s_validWords;
    int s_n_validWords;

    SimilarValss(){
        s_cover = nullptr;
        s_validWords = nullptr;
        s_coversize = 0;
        s_error = 0;
        s_n_validWords = 0;
    }

    void free() const{
        delete[] s_cover;
        delete[] s_validWords;
    }
};

class Search_cover_cache : public Search_base {
public:
    Search_cover_cache (NodeDataManager *nodeDataManager,
                        bool infoGain,
                        bool infoAsc,
                        bool repeatSort,
                        Support minsup,
                        Depth maxdepth,
                        int timeLimit,
                        Cache *cache,
                        float maxError = NO_ERR,
                        bool specialAlgo = true,
                        bool stopAfterError = false,
                        bool similarlb = false,
                        bool dynamic_branching = false,
                        bool similar_for_branching = true,
                        bool from_cpp = true);

    ~Search_cover_cache();

    void run ();

    bool similarlb;
    bool dynamic_branching;
    bool similar_for_branching;


private:
    pair<Node*,HasInter> recurse ( Itemset &itemset, Attribute last_added, Node* node, bool node_is_new, Attributes &attributes_to_visit, Depth depth, Error ub, SimilarValss &sim_db1, SimilarValss &sim_db2);
    Attributes getSuccessors(Attributes &last_freq_attributes, Attribute last_added);
    float informationGain (ErrorVals notTaken, ErrorVals taken);
    Node *getSolutionIfExists(Node *node, Error ub, Depth depth);
    Node* inferSolutionFromLB(Node *node, Error ub);
    Error computeSimilarityLB(SimilarValss &similar_db1, SimilarValss &similar_db2);
    void updateSimilarLBInfo1(NodeData *node_data, SimilarValss &highest_error_db, SimilarValss &highest_coversize_db);
    void updateSimilarLBInfo2(NodeData *node_data, SimilarValss &similar_db1, SimilarValss &similar_db2);
};

#endif