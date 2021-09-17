#include "cache_trie.h"
#include "rCoverWeighted.h"
#include "rCoverTotalFreq.h"
#include <algorithm>

using namespace std;

bool lessTrieEdge(const TrieEdge edge, const Item item) {
    return edge.item < item;
}

bool lessEdgeItem(const TrieEdge edge1, const TrieEdge edge2) {
    return edge1.item < edge2.item;
}

Cache_Trie::Cache_Trie(Depth maxdepth, WipeType wipe_type, int maxcachesize) : Cache(maxdepth, wipe_type, maxcachesize) {
    root = new TrieNode;
    cachesize = 0;
}

//look for itemset in the trie from root. Return null if not exist and the node of the last item if it exists
Node *Cache_Trie::get(Array<Item> itemset) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != itemset[i]) return nullptr; // item not found so itemset not found
        else cur_node = geqEdge_it->subtrie;
    }
    return cur_node;
}

void count_trie_size(const TrieNode *node, int &sum) {
    for (const auto &edge: node->edges) {
        sum += 1;
        count_trie_size(edge.subtrie, sum);
    }
}

int Cache_Trie::getCacheSize() {
    int size = 1;
    count_trie_size((TrieNode *) root, size);
    return size;
}

// classic top down
TrieNode *Cache_Trie::addNonExistingItemsetPart(Array<Item> itemset, int pos, vector<TrieEdge>::iterator &geqEdge_it,
                                                TrieNode *&cur_node,
                                                NodeDataManager *nodeDataManager) {
    for (int i = pos; i < itemset.size; ++i) {
        auto node = new TrieNode;
        TrieEdge newedge;
        newedge.item = itemset[i];
        newedge.subtrie = node;
        if (i == pos) cur_node->edges.insert(geqEdge_it, newedge);
        else { // new node added so add the edge without checking its place
            cur_node->edges.reserve(1);
            cur_node->edges.push_back(newedge);
        }
        cachesize++;
        cur_node = node;
    }
    cur_node->data = nodeDataManager->initData();
    return cur_node;
}

// insert itemset. Check from root and insert items only if they do not exist using addItemsetPart function
pair<Node *, bool> Cache_Trie::insert(Array<Item> itemset, NodeDataManager *nodeDataManager) {
    auto *cur_node = (TrieNode *) root;
    if (itemset.size == 0) {
        cachesize++;
        cur_node->data = nodeDataManager->initData();
        return {cur_node, true};
    }

    if (cachesize >= maxcachesize && maxcachesize > 0) {
//        cout << "cachesize before = " << cachesize << endl;
        wipe((TrieNode *) root, wipe_type);
        cachesize = getCacheSize();
//        cout << "cachesize after = " << cachesize << endl;
        if (cachesize >= maxcachesize) canwipe = false;
    }

    vector<TrieEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != itemset[i]) { // the item does not exist
            // create path representing the part of the itemset not yet present in the trie.
            TrieNode *last_inserted_node = addNonExistingItemsetPart(itemset, i, geqEdge_it, cur_node, nodeDataManager);
            return {last_inserted_node, true};
        } else {
            cur_node = geqEdge_it->subtrie;
            cur_node->count_opti_path++;
        }
    }
    bool is_newnode = cur_node->data == nullptr;
    if (is_newnode) cur_node->data = nodeDataManager->initData();
    return {cur_node, is_newnode};
}

#define effortCondition edge_iterator->subtrie->solution_effort < 10 * sqrt(max_solution_effort)
#define depthCondition depth == maxdepth
#define depthEffortCondition depth == maxdepth && effortCondition

void Cache_Trie::wipe(Node *node1, WipeType wipe_type, Depth depth) {
    auto *node = (TrieNode *) node1;
    for (auto edge_iterator = node->edges.begin(); edge_iterator != node->edges.end(); ++edge_iterator) {
        // recursively
        if (wipe_type == WipeEffort || wipe_type == WipeAll) wipe(edge_iterator->subtrie, wipe_type);
        else if (wipe_type == WipeDepth || wipe_type == WipeDepthEffort) wipe(edge_iterator->subtrie, wipe_type, depth + 1);

        // remove based on the right policy
        if (edge_iterator->subtrie->count_opti_path == 0) {
            if ((wipe_type == WipeAll) ||
                (wipe_type == WipeEffort && effortCondition) ||
                (wipe_type == WipeDepth && depthCondition) ||
                (wipe_type == WipeDepthEffort && depthEffortCondition)) {
                delete edge_iterator->subtrie;
                node->edges.erase(edge_iterator);
                --edge_iterator;
            }
        }
    }
}

void Cache_Trie::printload(TrieNode *node, vector<Item> &itemset) {
    for (auto & edge : node->edges) {
        itemset.push_back(edge.item);
        printload(edge.subtrie, itemset);
        itemset.pop_back();
    }
}

void Cache_Trie::updateItemsetLoad(Array<Item> itemset, bool inc) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end() && geqEdge_it->item == itemset[i]) { // item found
            cur_node = geqEdge_it->subtrie;
            if (inc) cur_node->count_opti_path++;
            else cur_node->count_opti_path--;
        }
    }
}

void Cache_Trie::updateSubTreeLoad(Array<Item> itemset, Item firstItem, Item secondItem, bool inc) {
    for (auto item: {firstItem, secondItem}) {
        if (item == -1) continue;
        Array<Item> itemset1 = addItem(itemset, item);
        if (item == secondItem) itemset.free();
        updateItemsetLoad(itemset1, inc);

        auto *node = (TrieNode *) get(itemset1);

        if (node && ((FND) node->data)->left && ((FND) node->data)->right) {
            Item nextFirstItem = item(((FND) node->data)->test, 0);
            Item nextSecondItem = item(((FND) node->data)->test, 1);
            updateSubTreeLoad(itemset1, nextFirstItem, nextSecondItem, inc);
        } else if (item == secondItem) itemset1.free();
    }
}

// original function
/*TrieNode *Cache_Trie::addNonExistingItemsetPart(Array<Item> itemset, int pos, TrieNode *&last_inserted_node, NodeDataManager* nodeDataManager) { // the items in itemset are inserted bottom up
    // remain_nodes_root is the node to concatenated to the last existing
    TrieNode *remain_nodes_root;
    // last_inserted_node is used to keep track of the last node corresponding to the last item in the itemset
    last_inserted_node = remain_nodes_root = new TrieNode;
    cachesize++;
    cout << "ici " << pos << endl;
    printItemset(itemset, true);
    for (int i = itemset.size - 2; i >= pos; --i) { // never enter in this loop when adding only one item
        cout << "jamais" << endl;
        TrieEdge newedge;
        newedge.item = itemset[i + 1];
        newedge.subtrie = remain_nodes_root;
        remain_nodes_root = new TrieNode;
        cachesize++;
        remain_nodes_root->edges.reserve(1); // assume that this is common
        remain_nodes_root->edges.push_back(newedge);
    }
    return remain_nodes_root;
}*/

// classic bottom up
/*TrieNode *Cache_Trie::addNonExistingItemsetPart(Array<Item> itemset, int pos, vector<TrieEdge>::iterator &geqEdge_it, TrieNode *&cur_node, NodeDataManager* nodeDataManager) { // the items in itemset are inserted bottom up
    // remain_nodes_root is the node to concatenated to the last existing
    TrieNode *remain_nodes_root, *last_inserted_node;
    // last_inserted_node is used to keep track of the last node corresponding to the last item in the itemset
    last_inserted_node = remain_nodes_root = new TrieNode;
    cachesize++;
//    cout << "ici " << pos << endl;
//    printItemset(itemset, true);
    for (int i = itemset.size - 2; i >= pos; --i) { // never enter in this loop when adding only one item
//        cout << "jamais" << endl;
        TrieEdge newedge;
        newedge.item = itemset[i + 1];
        newedge.subtrie = remain_nodes_root;
        remain_nodes_root = new TrieNode;
        cachesize++;
        remain_nodes_root->edges.reserve(1); // assume that this is common
        remain_nodes_root->edges.push_back(newedge);
    }

    TrieEdge newedge;
    newedge.item = itemset[pos];
    newedge.subtrie = remain_nodes_root;
    cur_node->edges.insert(geqEdge_it, newedge); // no need to sort during the search for elt since insert make it placed at the right position
    last_inserted_node->data = nodeDataManager->initData();
    return last_inserted_node;
}*/

// top down with each node initialize
/*template<typename Base, typename T>
inline bool instanceof(const T *) {
    return is_base_of<Base, T>::value;
}
TrieNode *Cache_Trie::addNonExistingItemsetPart(Array<Item> itemset, int pos, vector<TrieEdge>::iterator &geqEdge_it,
                                                TrieNode *&cur_node,
                                                NodeDataManager *nodeDataManager) { // the items in itemset are inserted bottom up
    if (pos == itemset.size - 1) { // just one item to add
        auto *node = new TrieNode;
        TrieEdge newedge;
        newedge.item = itemset[pos];
        newedge.subtrie = node;
        //auto geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[pos], lessTrieEdge);
        cur_node->edges.insert(geqEdge_it,
                               newedge); // no need to sort during the search for elt since insert make it placed at the right position
        cur_node->data = nodeDataManager->initData();
        cachesize++;
        return cur_node;
    } else {
        RCover *cover;
        if (instanceof<RCoverWeighted>(nodeDataManager->cover)) {
            cover = new RCoverWeighted(nodeDataManager->cover->dm,
                                       ((RCoverWeighted *) (nodeDataManager->cover))->weights);
        } else {
            cover = new RCoverTotalFreq(nodeDataManager->cover->dm); // non-weighted cover
        }
        auto s = cover->getSupportPerClass();
        forEachClass(n){
            cout << s[n] << ",";
        }
        cout << endl;
        cout << "exist: \n";
        for (int i = 0; i < pos; ++i) {
            cover->intersect(item_attribute(itemset[i]), item_value(itemset[i]) == 0 ? false : true);
            cout << itemset[i] << ": ";
            auto s = cover->getSupportPerClass();
            forEachClass(n){
                cout << s[n] << ",";
            }
            cout << endl;
        }
//        auto remain_nodes_root = new TrieNode;
        cout << "non-exist: \n";
        for (int i = pos; i < itemset.size; ++i) {
            cover->intersect(item_attribute(itemset[i]), item_value(itemset[i]) == 0 ? false : true);
            cout << itemset[i] << ": ";
            auto s = cover->getSupportPerClass();
            forEachClass(n){
                cout << s[n] << ",";
            }
            cout << endl;
//            auto node = (i == pos) ? remain_nodes_root : new TrieNode;
            auto node = new TrieNode;
            node->data = nodeDataManager->initData(cover);
            TrieEdge newedge;
            newedge.item = itemset[i];
            newedge.subtrie = node;
            if (i == pos) {
                cur_node->edges.insert(geqEdge_it,
                                       newedge); // no need to sort during the search for elt since insert make it placed at the right position
            } else {
                cur_node->edges.reserve(1); // assume that this is common
                cur_node->edges.push_back(newedge);
//                auto it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
//                cur_node->edges.insert(it, newedge); // no need to sort during the search for elt since insert make it placed at the right position
            }
            cachesize++;
            cur_node = node;
        }
        delete cover;
        return cur_node;
    }

}*/
