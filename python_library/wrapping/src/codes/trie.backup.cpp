#include trie_.h
#include "query_best.h"
#include <algorithm>

using namespace std;


Trie::Trie() {
    root = new TrieNode;
    root->data = nullptr;
}

TrieNode::~TrieNode() {
//    cout << "nodes deleted" << endl;
    if (data) delete data; //free (data); assumed allocated with malloc
    for (auto &edge : edges) delete edge.subtrie;
}

Trie::~Trie() {
//    cout << "root veut delete" << endl;
    delete root;
//    cout << "root est delete" << endl;
}

TrieNode *Trie::createTree(Array<Item> itemset, int pos, TrieNode *&last) {
    TrieNode *r2;
    last = r2 = new TrieNode;
    r2->data = nullptr;
    for (int i = itemset.size - 2; i >= pos; --i) { /// from
        TrieEdge newedge;
        newedge.item = itemset[i + 1];
        newedge.subtrie = r2;
        r2 = new TrieNode;
        r2->edges.reserve(1); // assume that this is common
        r2->edges.push_back(newedge);
        r2->data = nullptr;
    }
    return r2;
}

bool lessTrieEdge(const TrieEdge edge, const Item item) {
    return edge.item < item;
}

TrieNode *Trie::find(Array<Item> itemset) { ///seek itemset in the trie from root. Return null if not exist and the node of the last item if it exists
    TrieNode *p = root, *p2;
    vector<TrieEdge>::iterator t, e;

    forEach (i, itemset) {
        e = p->edges.end();
        t = lower_bound(p->edges.begin(), e, itemset[i], lessTrieEdge);
        if (t == e || t->item != itemset[i]) { // item not found
            return nullptr; // not found
        } else
            p = t->subtrie;
    }
    return p;
}

bool lessEdgeItem(const TrieEdge edge1, const TrieEdge edge2) {
    return edge1.item < edge2.item;
}

/// insert itemset. Check from root and insert items only they do not exist using createTree
TrieNode *Trie::insert(Array<Item> itemset) {
// check size before insert. remove an element to potentially liberate space first
// implement the remove and the pop. impement the remove from the trie
    TrieNode *p = root, *p2;
    vector<TrieEdge>::iterator t, e;

    forEach (i, itemset) {
//        sort(p->edges.begin(), p->edges.end(), lessEdgeItem);
        e = p->edges.end();
        t = lower_bound(p->edges.begin(), e, itemset[i], lessTrieEdge);
        if (t == e || t->item != itemset[i]) { /// if item does not exist
            // not found, insert
            TrieEdge newedge;
            newedge.item = itemset[i];
            p2 = p;
            /// create path representing the part of the itemset not yet present in the trie. So you have to provide
            /// the position at which the part not present starts and the last node at which we must complete the tree
            newedge.subtrie = createTree(itemset, i, p2);
            p->edges.insert(t, newedge); // no need to sort during the search for elt since insert make it placed at the right position

            return p2;
        } else {
            p = t->subtrie;
        }
    }
    return p;
}
