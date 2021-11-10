#ifndef SEARCH_CACHE_H
#define SEARCH_CACHE_H

#include "globals.h"
#include "cache.h"
#include "cache_trie.h"
#include "solution.h"
#include "search_base.h"

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

class Search_cache : public Search_base{
public:
    Search_cache (NodeDataManager *nodeDataManager,
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
                  bool dynamic_branching = false);

    ~Search_cache();

    void run ();

    Cache *cache;
    bool similarlb;
    bool dynamic_branching;


private:
    pair<Node*,HasInter> recurse ( Array<Item> itemset, Attribute last_added, Node* node, bool node_is_new, Array<Attribute> attributes_to_visit, Depth depth, Error ub);
    Array<Attribute> getSuccessors(Array<Attribute> last_freq_attributes, Attribute last_added, Node* node);
    float informationGain (ErrorVals notTaken, ErrorVals taken);
    Node *getSolutionIfExists(Node *node, Error ub, Depth depth);
    Error computeSimilarityLB(SimilarVals &similar_db1, SimilarVals &similar_db2);
    void updateSimilarLBInfo1(NodeData *node_data, SimilarVals &highest_error_db, SimilarVals &highest_coversize_db);
    void updateSimilarLBInfo2(NodeData *node_data, SimilarVals &similar_db1, SimilarVals &similar_db2);
};

#endif