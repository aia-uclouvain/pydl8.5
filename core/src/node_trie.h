//
// Created by Gael Aglin on 14/01/2022.
//

#ifndef DL85_NODE_TRIE_H
#define DL85_NODE_TRIE_H

#include "globals.h"
#include "node.h"
#include <vector>

struct TrieNode;

struct TrieEdge {
    Item item;
    TrieNode *subtrie;
};

struct TrieNode : Node {
    int n_subnodes = 0;
    int n_reuse = 0;
    Depth depth = 0;
    std::vector<TrieEdge> edges;
//    unordered_set<TrieNode*> search_parents;
//    unordered_set<pair<TrieNode*,Itemset>> search_parents;
    TrieNode* trie_parent;
    TrieFreq_NodeData *data; // data is the information kept by a node during the tree search

    TrieNode(): Node() {
        data = nullptr;
    }
    ~TrieNode() {
        delete data;
    }

};

#endif //DL85_NODE_TRIE_H
