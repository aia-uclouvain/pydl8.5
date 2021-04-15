#ifndef TRIE_H
#define TRIE_H
#include <vector> // we only use arrays for +- fixed sized things
#include "globals.h"
#include "query.h"
#include <queue>

using namespace std;

typedef pair<int, TrieNode*> NodePriority;

struct TrieNode;

struct TrieEdge {
  Item item;
  TrieNode *subtrie;
};

struct TrieNode {
  vector<TrieEdge> edges;
  QueryData *data; // data used to answer a query, if null this itemset is not closed
  ~TrieNode ();
};


class Trie {
friend class Query_TotalFreq;
public:
    Trie();

    ~Trie();
    TrieNode *insert ( Array<Item> itemset );
    TrieNode *find ( Array<Item> itemset );
    TrieNode *root;
    TrieNode *createTree ( Array<Item> itemset, int pos, TrieNode *&last );
    struct cmp {
        bool operator()(const NodePriority &a, const NodePriority &b) {
            return a.first > b.first; // < for Max heap (highest first) and > for min heap (lowest first)
        };
    };
    priority_queue<NodePriority, vector<NodePriority>, cmp> nodemapper;
    int maxcachesize;

};

#endif
