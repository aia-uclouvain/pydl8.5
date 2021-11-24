#include "cache_ltd_trie.h"
#include "rCoverWeight.h"
#include "rCoverFreq.h"
#include <algorithm>
#include "logger.h"

using namespace std;

bool lessTrieEdge(const TrieLtdEdge edge, const Item item) {
    return edge.item < item;
}

bool lessEdgeItem(const TrieLtdEdge edge1, const TrieLtdEdge edge2) {
    return edge1.item < edge2.item;
}

// sort the nodes in decreasing order in the aim to get the lowest in O(1) as it will be the last
// ascending order ===> return true if a < b and false if a > b
// descending order ===> return true if a > b and false if a < b
//bool sortDecOrder(const pair<TrieLtdNode*,vector<Item>> &pair1, const pair<TrieLtdNode*,vector<Item>> &pair2) {
bool sortDecOrder(const pair<TrieLtdNode*,TrieLtdNode*> &pair1, const pair<TrieLtdNode*,TrieLtdNode*> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->count_opti_path > 0 && node2->count_opti_path == 0) return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->count_opti_path == 0 && node2->count_opti_path > 0) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}

bool sortReuseDecOrder(const pair<TrieLtdNode*,TrieLtdNode*> &pair1, const pair<TrieLtdNode*,TrieLtdNode*> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->count_opti_path > 0 && node2->count_opti_path == 0) return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->count_opti_path == 0 && node2->count_opti_path > 0) return false; // same for the node2
    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth) return node1->depth < node2->depth; // depth from bigger to lower, so sorted in creasing order
//    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth) return node1->support > node2->support;
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}

// in case of heap decreasing comparator for sort provides a min-heap
//bool minHeapOrder(const pair<TrieLtdNode*,vector<Item>> &pair1, const pair<TrieLtdNode*,vector<Item>> &pair2) {
bool minHeapOrder(const pair<TrieLtdNode*,TrieLtdNode*> &pair1, const pair<TrieLtdNode*,TrieLtdNode*> &pair2) {
    return sortDecOrder(pair1, pair2);
}

Cache_Ltd_Trie::Cache_Ltd_Trie(Depth maxdepth, WipeType wipe_type, int maxcachesize, float wipe_factor) : Cache(maxdepth, wipe_type, maxcachesize), wipe_factor(wipe_factor) {
    root = new TrieLtdNode;
    if (maxcachesize > NO_CACHE_LIMIT) heap.reserve(maxcachesize);
}

// look for itemset in the trie from root. Return null if not exist and the node of the last item if it exists
Node *Cache_Ltd_Trie::get(Array<Item> itemset) {
    auto *cur_node = (TrieLtdNode *) root;
    vector<TrieLtdEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != itemset[i]) return nullptr; // item not found so itemset not found
        else cur_node = geqEdge_it->subtrie;
    }
    return cur_node;
}

int Cache_Ltd_Trie::getCacheSize() {
    if (maxcachesize == NO_CACHE_LIMIT) return cachesize;
    else return heap.size() + 1;
}

// classic top down
TrieLtdNode *Cache_Ltd_Trie::addNonExistingItemsetPart(Array<Item> itemset, int pos, vector<TrieLtdEdge>::iterator &geqEdge_it, TrieLtdNode *parent_node) {
    TrieLtdNode* child_node;
    for (int i = pos; i < itemset.size; ++i) {
        child_node = new TrieLtdNode();
        if (maxcachesize > NO_CACHE_LIMIT) heap.push_back(make_pair(child_node, parent_node));
        TrieLtdEdge newedge{itemset[i], child_node};
        if (i == pos) parent_node->edges.insert(geqEdge_it, newedge);
        else parent_node->edges.push_back(newedge); // new node added so add the edge without checking its place
        cachesize++;
        child_node->depth = i + 1;
        parent_node = child_node;
    }
    return child_node;
}

// insert itemset. Check from root and insert items only if they do not exist using addItemsetPart function
pair<Node*, bool> Cache_Ltd_Trie::insert(Array<Item> itemset) {
    auto *cur_node = (TrieLtdNode *) root;
    if (itemset.size == 0) {
        cachesize++;
        return {cur_node, true};
    }
    if (getCacheSize() >= maxcachesize && maxcachesize > 0) wipe();

    vector<TrieLtdEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != itemset[i]) { // the item does not exist
            // create path representing the part of the itemset not yet present in the trie.
            TrieLtdNode *last_inserted_node = addNonExistingItemsetPart(itemset, i, geqEdge_it, cur_node);
            return {last_inserted_node, true};
        } else {
            if (i == 0) cur_node->n_reuse++; // root node
            cur_node = geqEdge_it->subtrie;
            cur_node->count_opti_path++;
            cur_node->n_reuse++;
            cur_node->depth = i + 1;
        }
    }
    if (cur_node->data == nullptr) {
        return {cur_node, true};
    }
    else {
        Logger::showMessageAndReturn("The node already exists");
        return {cur_node, false};
    }
}

void wipeAll(TrieLtdNode *node) {
    for (auto edge_iterator = node->edges.begin(); edge_iterator != node->edges.end(); ++edge_iterator) {
        // as we don't want children without parent, we perform a postfix search and recursively remove all nodes not in potential optimal tree
        wipeAll(edge_iterator->subtrie);
        if (edge_iterator->subtrie->count_opti_path == 0) {
            delete edge_iterator->subtrie;
            node->edges.erase(edge_iterator);
            --edge_iterator;
        }
    }
}

void Cache_Ltd_Trie::wipe() {
    int n_del = (int) (maxcachesize * wipe_factor);

    /*if (wipe_type == All){ // the problem here is that the priority queue still contain the removed nodes
        wipeAll((TrieLtdNode*)root);
        return;
    }*/

    switch (wipe_type) {
        case Subnodes:
            computeSubNodes((TrieLtdNode*)root);
            // cout << "is subnodes hierarchy consistent : " << isConsistent((TrieLtdNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            break;
        case Recall:
            // cout << "is reuse hierarchy consistent : " << isConsistent((TrieLtdNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortReuseDecOrder);
            break;
        default: // All. this block is used when the above if is commented.
            computeSubNodes((TrieLtdNode*)root);
            // cout << "is subnodes hierarchy consistent for all wipe : " << isConsistent((TrieLtdNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            n_del = heap.size();
    }

//     cout << "cachesize before wipe = " << getCacheSize() << endl;
    int counter = 0;
    for (auto it = heap.rbegin(); it != heap.rend(); it++) {
        if (counter == n_del || (*it).first->count_opti_path > 0) break;
        // remove the edge bringing to the node
        it->second->edges.erase(find_if(it->second->edges.begin(), it->second->edges.end(), [it](const TrieLtdEdge &look) { return look.subtrie == it->first; }));
        // remove the node
        delete it->first;
        heap.pop_back();
        counter++;
    }
    // cout << "cachesize after wipe = " << getCacheSize() << endl;

    /*make_heap(heap.begin(), heap.end(), minHeapOrder);
    for (int i = 0; i < n_del; ++i) {
        if (heap.front().first->count_opti_path > 0) break;
        pop_heap(heap.begin(), heap.end(), minHeapOrder);
        delete heap.back().first;
        heap.back().second->edges.erase(find_if(heap.back().second->edges.begin(), heap.back().second->edges.end(), [this](const TrieLtdEdge &look){ return look.subtrie == this->heap.back().first; }));
        heap.pop_back();
    }*/
}

int Cache_Ltd_Trie::computeSubNodes(TrieLtdNode* node) {
    if (node->edges.empty()) return 0;
    node->n_subnodes = 0;
    for (auto edge: node->edges) node->n_subnodes += 1 + computeSubNodes(edge.subtrie);
    return node->n_subnodes;
}



bool Cache_Ltd_Trie::isConsistent(TrieLtdNode* node){
    return all_of(node->edges.begin(), node->edges.end(), [node,this] (const TrieLtdEdge &edge) {
        TrieLtdNode *parent = node, *child = edge.subtrie;
        if(
                ((wipe_type == Subnodes || wipe_type == All) && child->n_subnodes >= parent->n_subnodes) ||
                (wipe_type == Recall && (child->n_reuse > parent->n_reuse || (child->n_reuse == parent->n_reuse && child->depth <= parent->depth)))
          ) return false;
        else { bool res = isConsistent(child); if (not res) return false; }
        return true;
    });
}

/*bool Cache_Ltd_Trie::isConsistent(TrieLtdNode* node){
    TrieLtdNode *parent = node;
    for (auto edge: node->edges) {
        TrieLtdNode *child = edge.subtrie;
        if(
                (wipe_type == Subnodes && child->n_subnodes >= parent->n_subnodes) ||
                (wipe_type == Recall && (child->n_reuse > parent->n_reuse || (child->n_reuse == parent->n_reuse && child->depth <= parent->depth)))
          ) return false;
        else { bool res = isConsistent(child); if (not res) return false; }
        return true;
    }
    return true;
}*/

void Cache_Ltd_Trie::updateItemsetLoad(Array<Item> itemset, bool inc) {
    auto *cur_node = (TrieLtdNode *) root;
    vector<TrieLtdEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end() && geqEdge_it->item == itemset[i]) { // item found
            cur_node = geqEdge_it->subtrie;
            if (inc) cur_node->count_opti_path++;
            else cur_node->count_opti_path--;
//            if (cur_node->count_opti_path < 0) cout << "load = " << cur_node->count_opti_path << endl;
        }
    }
}

void Cache_Ltd_Trie::updateSubTreeLoad(Array<Item> itemset, Item firstItem, Item secondItem, bool inc) {
    for (auto item: {firstItem, secondItem}) {
        if (item == -1) continue;
        Array<Item> itemset1 = addItem(itemset, item);
        if (item == secondItem) itemset.free();
        updateItemsetLoad(itemset1, inc);

        auto *node = (TrieLtdNode *) get(itemset1);

        if (node and node->data and ((FND) node->data)->left and ((FND) node->data)->right) {
            Item nextFirstItem = item(((FND) node->data)->test, 0);
            Item nextSecondItem = item(((FND) node->data)->test, 1);
            updateSubTreeLoad(itemset1, nextFirstItem, nextSecondItem, inc);
        } else if (item == secondItem) itemset1.free();
    }
}

void Cache_Ltd_Trie::updateRootPath(Array<Item> itemset, int value) {
    auto *cur_node = (TrieLtdNode *) root;
    vector<TrieLtdEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        cur_node = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge)->subtrie;
        cur_node->count_opti_path += value;
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
