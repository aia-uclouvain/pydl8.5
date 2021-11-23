#ifndef SEARCH_HASH_COVER_H
#define SEARCH_HASH_COVER_H

#include "globals.h"
#include "cache_hash_cover.h"
#include "solution.h"
#include "search_base.h"

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

class Search_hash_cover : public Search_base{
public:
    Search_hash_cover (NodeDataManager *nodeDataManager,
                  bool infoGain,
                  bool infoAsc,
                  bool repeatSort,
                  Support minsup,
                  Depth maxdepth,
                  Cache *cache,
                  int timeLimit,
                  float maxError = NO_ERR,
                  bool specialAlgo = true,
                  bool stopAfterError = false,
                  bool similarlb = false,
                  bool dynamic_branching = false,
                  bool similar_for_branching = true);

    ~Search_hash_cover();

    void run ();

    Cache *cache;
    bool similarlb;
    bool dynamic_branching;
    bool similar_for_branching;


private:
    pair<Node*,HasInter> recurse ( Array<Item> itemset, Attribute last_added, Node* node, bool node_is_new, Array<Attribute> attributes_to_visit, Depth depth, Error ub, SimilarValss &sim_db1, SimilarValss &sim_db2);
    Array<Attribute> getSuccessors(Array<Attribute> last_freq_attributes, Attribute last_added, Node* node);
    float informationGain (ErrorVals notTaken, ErrorVals taken);
    Node *getSolutionIfExists(Node *node, Error ub, Depth depth);
    Node* inferSolutionFromLB(Node *node, Error ub);
    Error computeSimilarityLB(SimilarValss &similar_db1, SimilarValss &similar_db2);
    void updateSimilarLBInfo1(NodeData *node_data, SimilarValss &highest_error_db, SimilarValss &highest_coversize_db);
    void updateSimilarLBInfo2(NodeData *node_data, SimilarValss &similar_db1, SimilarValss &similar_db2);
};

#endif