#ifndef TRIE_H
#define TRIE_H
#include <vector> // we only use arrays for +- fixed sized things
#include "globals.h"
#include "query.h"
#include <queue>

using namespace std;

typedef pair<int, TrieNode*> NodePriority;
typedef int hashcode;

//struct TrieNode;

struct TrieNode {
    Array<Item> itemset;
//    vector<hashcode> children;
//    vector<hashcode> parent;
    QueryData *data; // data used to answer a query, if null this itemset is not closed
     ~TrieNode ();
};

class Trie {
friend class Query_TotalFreq;
public:
    Trie(int maxsize);
    ~Trie();

    int cachesize;
    TrieNode** bucket;
    TrieNode *root;
    TrieNode *insert ( Array<Item> itemset );
    hashcode gethashcode ( Array<Item> itemset );
    hashcode getlasthashcode ( Array<Item> itemset );
    void remove( hashcode code );

    /*TrieNode *find ( Array<Item> itemset );
    TrieNode *root;
    TrieNode *createTree ( Array<Item> itemset, int pos, TrieNode *&last );
    struct cmp {
        bool operator()(const NodePriority &a, const NodePriority &b) {
            return a.first > b.first; // < for Max heap (highest first) and > for min heap (lowest first)
        };
    };
    priority_queue<NodePriority, vector<NodePriority>, cmp> nodemapper;
    int maxcachesize;*/

};

#endif
