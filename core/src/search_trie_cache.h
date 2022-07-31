#ifndef SEARCH_TRIE_CACHE_H
#define SEARCH_TRIE_CACHE_H

#include "search_base.h"
#include "nodeDataManager_Trie.h"
#include <chrono>

typedef bool HasInter;

struct SimilarVals{
    bitset<M>* s_cover;
    Support s_coversize;
    Error s_error;
    int* s_validWords;
    int s_n_validWords;

    SimilarVals(){
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

class Search_trie_cache : public Search_base{
public:
    Search_trie_cache (NodeDataManager *nodeDataManager,
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

    ~Search_trie_cache();

    void run ();

    bool similarlb;
    bool dynamic_branching;
    bool similar_for_branching;


private:
    pair<Node*,HasInter> recurse ( const Itemset &itemset, Attribute last_added, Node* node, bool node_is_new, Attributes &attributes_to_visit, Depth depth, Error ub, SimilarVals &sim_db1, SimilarVals &sim_db2);
    Attributes getSuccessors(Attributes &last_freq_attributes, Attribute last_added, const Itemset &itemset, Node*);
    float informationGain (ErrorVals notTaken, ErrorVals taken);
    Node *getSolutionIfExists(Node *node, Error ub, Depth depth, const Itemset &itemset);
    Node* inferSolutionFromLB(Node *node, Error ub);
    Error computeSimilarityLB(SimilarVals &similar_db1, SimilarVals &similar_db2, bool quiet = true);
    bool updateSimilarLBInfo1(NodeData *node_data, SimilarVals &highest_error_db, SimilarVals &highest_coversize_db);
    bool updateSimilarLBInfo2(NodeData *node_data, SimilarVals &similar_db1, SimilarVals &similar_db2);
    void retrieveWipedSubtrees(Node *node, const Itemset &itemset, Item last_added, Attributes &attributes, Depth depth);
    bool isTreeComplete(Node* node, const Itemset &itemset);
};

#endif