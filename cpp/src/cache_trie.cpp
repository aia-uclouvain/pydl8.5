#include "cache_trie.h"
#include <algorithm>

using namespace std;

bool lessTrieEdge(const TrieEdge edge, const Item item) {
    return edge.item < item;
}

bool lessEdgeItem(const TrieEdge edge1, const TrieEdge edge2) {
    return edge1.item < edge2.item;
}


Cache_Trie::Cache_Trie(int maxcachesize): maxcachesize(maxcachesize), Cache() {
    root = new TrieNode;
    cachesize = 0;
}

TrieNode *Cache_Trie::createTree(Array<Item> itemset, int pos, TrieNode *&last, NodeDataManager* nodeDataManager) {
    TrieNode *r2;
    last = r2 = new TrieNode;
//    if (!r2->data) r2->data = nodeDataManager->initData();
    cachesize++;
    for (int i = itemset.size - 2; i >= pos; --i) { // never enter in this loop when adding only one item
        TrieEdge newedge;
        newedge.item = itemset[i + 1];
        newedge.subtrie = r2;
        r2 = new TrieNode;
//        if (!r2->data) r2->data = nodeDataManager->initData();
        cachesize++;
        r2->edges.reserve(1); // assume that this is common
        r2->edges.push_back(newedge);
//        cachesize++;
    }
    return r2;
}

void Cache_Trie::updateItemsetLoad(Array<Item> itemset, bool inc) { ///seek itemset in the trie from root. Return null if not exist and the node of the last item if it exists
    TrieNode *p = (TrieNode*)root, *p2;
    vector<TrieEdge>::iterator t, e;

    forEach (i, itemset) {
        e = p->edges.end();
        t = lower_bound(p->edges.begin(), e, itemset[i], lessTrieEdge);
        if (t == e || t->item != itemset[i]) { // item not found
            return; // not found
        } else {
            p = t->subtrie;
            if (inc) p->count_opti_path++;
            else p->count_opti_path--;
        }
    }
}

Node *Cache_Trie::get(Array<Item> itemset) { ///seek itemset in the trie from root. Return null if not exist and the node of the last item if it exists
    TrieNode *p = (TrieNode*)root, *p2;
    vector<TrieEdge>::iterator t, e;

//    if (item != -1) itemset = addItem(itemset, item);
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

/*void count_trie_size(TrieNode* node, int& sum){
    if (node->edges.empty()){
        sum += 1;
        return ;
    }
    for (auto &edge: node->edges) {
        count_trie_size(edge.subtrie, sum);
        sum += 1;
    }
}*/

void count_trie_size(TrieNode* node, int& sum){
    for (auto &edge: node->edges) {
        sum += 1;
        count_trie_size(edge.subtrie, sum);
    }
}

/// insert itemset. Check from root and insert items only they do not exist using createTree
pair<Node *, bool> Cache_Trie::insert(Array<Item> itemset, NodeDataManager* nodeDataManager) {
    ///cout << "\nitemset: "; printItemset(itemset, true);
    if (itemset.size == 0){
        cachesize++;
    }
    if (cachesize >= maxcachesize && maxcachesize > 0) {
        vector<Item> v;
        int unsure_count = 0, inopti_count = 1;
        wipe((TrieNode*)root, v, unsure_count, inopti_count);
        cachesize = 1;
        count_trie_size((TrieNode*)root, cachesize);
//        cachesize++;
        if (cachesize >= maxcachesize) canwipe = false;
    }
    TrieNode *p = (TrieNode*)root, *p2;
    vector<TrieEdge>::iterator t, e;
    bool newnode;

    forEach (i, itemset) {
        e = p->edges.end();
        //lb: first elt >= key //ub: first elt > key //otherwise: end()
        t = lower_bound(p->edges.begin(), e, itemset[i], lessTrieEdge);
        if (t == e || t->item != itemset[i]) { /// if item does not exist
//            cachesize++;
            TrieEdge newedge;
            newedge.item = itemset[i];
            p2 = p;
            /// create path representing the part of the itemset not yet present in the trie. So you have to provide
            /// the position at which the part not present starts and the last node at which we must complete the tree
            newedge.subtrie = createTree(itemset, i, p2, nodeDataManager);
            p->edges.insert(t, newedge); // no need to sort during the search for elt since insert make it placed at the right position

            newnode = p2->data ? false : true;
            if (newnode) p2->data = nodeDataManager->initData();
            return {p2, newnode};
        } else {
            p = t->subtrie;
            p->count_opti_path++;
        }
    }
    newnode = p->data ? false : true;
    if (newnode) p->data = nodeDataManager->initData();
    return {p, newnode};
}

/*void Cache_Trie::wipe(TrieNode* node, vector<Item>& itemset, int &unsure_count, int &inopti_count) {
//    cout << "\nItemset to explore: ";
//    if (itemset.empty()) cout << "phi";
//    else for (auto item : itemset) cout << item << ", ";
//    cout << endl;
//    cout << "size: " << node->edges.size() << endl;
//
//    if (itemset.size() == 2 && itemset[0] == 1 && itemset[1] == 2){
//        for (auto edge: node->edges) cout << edge.item << ", ";
//        cout << endl;
//    }

//    vector<Item> v;
    for (auto edge = node->edges.begin(); edge != node->edges.end(); ++edge){
        itemset.push_back(edge->item);
        wipe(edge->subtrie, itemset, unsure_count, inopti_count);
        if (edge->subtrie->load == 0) {
//            if (itemset.size() == 2 && itemset[0] == 31 && itemset[1] == 42) cout << "SUPPP" << endl;
//            cout << "Itemset: ";
//            for (auto item : itemset) cout << item << ", ";
//            cout << endl;
            //cout << edge->item << endl;
            //cout << "Error: " << ((Freq_NodeData*)edge->subtrie->data) << endl;
            //cout << "Error: " << ((Freq_NodeData*)edge->subtrie->data)->leafError << endl;
            //cout << endl;
//            if (edge->subtrie) delete ((TrieNode*)edge->subtrie);
            delete edge->subtrie;
            node->edges.erase(edge);
            --edge;
//            --cachesize;
//            v.push_back(edge->item);
//            cout << "here" << endl << endl;
        } //else break;
        else if (edge->subtrie->load == -1) unsure_count++;
        else inopti_count++;
        itemset.pop_back();
    }
}*/

void Cache_Trie::wipe(TrieNode* node, vector<Item>& itemset, int &unsure_count, int &inopti_count) {
    for (auto edge = node->edges.begin(); edge != node->edges.end(); ++edge){
        itemset.push_back(edge->item);
        wipe(edge->subtrie, itemset, unsure_count, inopti_count);
        if (edge->subtrie->count_opti_path == 0) {
            delete edge->subtrie;
            node->edges.erase(edge);
            --edge;
//            --cachesize;
//            v.push_back(edge->item);
        } //else break;
        else if (edge->subtrie->count_opti_path < 0) for(;;) cout << "";
        else inopti_count++;
        itemset.pop_back();
    }
}

void Cache_Trie::printload(TrieNode* node, vector<Item>& itemset) {

    for (auto edge = node->edges.begin(); edge != node->edges.end(); ++edge){
        itemset.push_back(edge->item);
        printload(edge->subtrie, itemset);
        itemset.pop_back();
    }
}

void Cache_Trie::updateSubTreeLoad(Array<Item> itemset, Item firstItem, Item secondItem, bool inc) {

    for (auto item: {firstItem, secondItem}) {
        if (item == -1) continue;
        Array<Item> itemset1 = addItem(itemset, item);
        if (item == secondItem) itemset.free();
        updateItemsetLoad(itemset1, inc);

        auto* node = (TrieNode*)get(itemset1);

        if ( ((FND)node->data)->left && ((FND)node->data)->right ){
            Item nextFirstItem = item(((FND)node->data)->test, 0);
            Item nextSecondItem = item(((FND)node->data)->test, 1);
            updateSubTreeLoad( itemset1, nextFirstItem, nextSecondItem, inc );
        }
        else if (item == secondItem) itemset1.free();
    }

}
