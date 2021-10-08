#include "cache_ltd_trie.h"
#include "rCoverWeighted.h"
#include "rCoverTotalFreq.h"
#include <algorithm>

using namespace std;

bool lessTrieEdge(const TrieLtdEdge edge, const Item item) {
    return edge.item < item;
}

// in case of heap a < b creates max-heap
bool minHeapOrder(const TrieLtdNode &node1, const TrieLtdNode &node2) {
//    return node1.solution_effort < node2.solution_effort; // max-heap (high to low)
    return node1.solution_effort > node2.solution_effort; // min-heap (low to high)
}

// in case of algorithm sort, a < b provides ascending order
/*bool decNodeOrder(const TrieLtdNode &node1, const TrieLtdNode &node2) {
    return node1.solution_effort < node2.solution_effort; // max-heap (high to low)
//    return node1.solution_effort > node2.solution_effort; // min-heap (low to high)
}*/

/*bool sortDecOrder(const TrieLtdNode &node1, const TrieLtdNode &node2) {

    if (node1.count_opti_path > 0) return true; // place node1 to left (high value)
    if (node1.count_opti_path < 0) return false; // place invalid node1 to right to be popped first

    if (node2.count_opti_path > 0) return false; // place node2 to left (high value)
    if (node2.count_opti_path < 0) return true; // place invalid node2 to right to be popped first

    if (node1.solution_effort == node2.solution_effort) return ((const Freq_NodeData*)node1.data)->test > ((const Freq_NodeData*)node2.data)->test;

    return node1.solution_effort > node2.solution_effort;
}*/

// decreasing order for algorithm sort
// ascending order ===> true if a < b and false if a > b
// descending order ===> true if a > b and false if a < b
// a < b ===> ascending order
// a > b ===> descending order
//bool sortDecOrder(pair<TrieLtdNode *, TrieLtdNode*> &pair1, pair<TrieLtdNode *, TrieLtdNode*> &pair2) {
/*bool sortDecOrder(tuple<TrieLtdNode *, TrieLtdNode*, Item, vector<Item>> &pair1, tuple<TrieLtdNode *, TrieLtdNode*, Item, vector<Item>> &pair2) {
//const auto node1 = pair1.first;
const auto node1 = get<0>(pair1);
//const auto node2 = pair2.first;
const auto node2 = get<0>(pair2);

    if ((node1->count_opti_path < 0 && node2->count_opti_path < 0) || (node1->data == nullptr && node2->data == nullptr)) return false; // use node1 as the lowest
    if (node1->count_opti_path > 0 && node2->count_opti_path > 0) return true; // use node1 as the highest
    // true if node1 > node2 or node2 < node1
    if (node1->count_opti_path > 0 || node2->count_opti_path < 0 || node2->data == nullptr) return true; // place node1 to left (high value)
    // false if node1 < node2 or node2 > node1
    if (node1->count_opti_path < 0 || node2->count_opti_path > 0 || node1->data == nullptr) return false; // place invalid node1 to right to be popped first

    // in case we don't care about the ub impact for nodes without solution
    if (node1->solution_effort == node2->solution_effort) return ((const FND)node1->data)->test > ((const FND)node2->data)->test;
    return node1->solution_effort > node2->solution_effort;

    if (node1->solution_effort == node2->solution_effort && ((const FND)node1->data)->test != ((const FND)node2->data)->test) {
        // if a return statement were used here, in case of tests equality node1 will be used as higher without using ub impact
        if (((const FND)node1->data)->test > ((const FND)node2->data)->test) return true;
        if (((const FND)node1->data)->test < ((const FND)node2->data)->test) return false;
    }
    float n1_eff = ((const FND)node1->data)->test != -1 ? node1->solution_effort : ((const FND)node1->data)->leafError / ((const FND)node1->data)->lowerBound * node1->solution_effort;
    float n2_eff = ((const FND)node2->data)->test != -1 ? node2->solution_effort : ((const FND)node2->data)->leafError / ((const FND)node2->data)->lowerBound * node2->solution_effort;

    return n1_eff > n2_eff;
}*/

bool sortDecOrder(tuple<TrieLtdNode *, TrieLtdNode*, Item, vector<Item>> &pair1, tuple<TrieLtdNode *, TrieLtdNode*, Item, vector<Item>> &pair2) {
//const auto node1 = pair1.first;
    const auto node1 = get<0>(pair1);
//const auto node2 = pair2.first;
    const auto node2 = get<0>(pair2);

//    if ((node1->count_opti_path < 0 && node2->count_opti_path < 0) || (node1->data == nullptr && node2->data == nullptr)) return false; // use node1 as the lowest
    //if (node1->count_opti_path < 0 && node2->count_opti_path < 0) return false; // use node1 as the lowest
    //if (node1->count_opti_path > 0 && node2->count_opti_path > 0) return true; // use node1 as the highest
    // true if node1 > node2 or node2 < node1
//    if (node1->count_opti_path > 0 || node2->count_opti_path < 0 || node2->data == nullptr) return true; // place node1 to left (high value)
    //if (node1->count_opti_path > 0 || node2->count_opti_path < 0) return true; // place node1 to left (high value)
    // false if node1 < node2 or node2 > node1
//    if (node1->count_opti_path < 0 || node2->count_opti_path > 0 || node1->data == nullptr) return false; // place invalid node1 to right to be popped first
    //if (node1->count_opti_path < 0 || node2->count_opti_path > 0) return false; // place invalid node1 to right to be popped first

    if (node1->count_opti_path > 0 && node2->count_opti_path == 0) return true; // place node1 to left (high value)
    if (node1->count_opti_path == 0 && node2->count_opti_path > 0) return false; // place invalid node1 to right to be popped first

    // in case we don't care about the ub impact for nodes without solution
    //if (node1->n_subnodes == node2->n_subnodes) return ((const FND)node1->data)->test > ((const FND)node2->data)->test;
    return node1->n_subnodes > node2->n_subnodes;

    if (node1->solution_effort == node2->solution_effort && ((const FND)node1->data)->test != ((const FND)node2->data)->test) {
        // if a return statement were used here, in case of tests equality node1 will be used as higher without using ub impact
        if (((const FND)node1->data)->test > ((const FND)node2->data)->test) return true;
        if (((const FND)node1->data)->test < ((const FND)node2->data)->test) return false;
    }
    float n1_eff = ((const FND)node1->data)->test != -1 ? node1->solution_effort : ((const FND)node1->data)->leafError / ((const FND)node1->data)->lowerBound * node1->solution_effort;
    float n2_eff = ((const FND)node2->data)->test != -1 ? node2->solution_effort : ((const FND)node2->data)->leafError / ((const FND)node2->data)->lowerBound * node2->solution_effort;

    return n1_eff > n2_eff;
}

bool lessEdgeItem(const TrieLtdEdge edge1, const TrieLtdEdge edge2) {
    return edge1.item < edge2.item;
}

Cache_Ltd_Trie::Cache_Ltd_Trie(Depth maxdepth, WipeType wipe_type, int maxcachesize) : Cache(maxdepth, wipe_type, maxcachesize) {
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
TrieLtdNode *Cache_Ltd_Trie::addNonExistingItemsetPart(Array<Item> itemset, int pos, vector<TrieLtdEdge>::iterator &geqEdge_it,
                                                TrieLtdNode *parent_node,
                                                NodeDataManager *nodeDataManager) {
    TrieLtdNode* child_node;
    int sub_nodes = itemset.size - pos - 1;
    for (int i = pos; i < itemset.size; ++i) {
        child_node = new TrieLtdNode();
        if (i != itemset.size) child_node->n_subnodes = sub_nodes;
//        cout << "node inserted: ";
//        for (int z = 0; z <= i; ++z) {
//            cout << itemset[z] << ",";
//        }
//        cout << " pointer is " << child_node << " its parent " << parent_node << endl;
//        if (maxcachesize > NO_CACHE_LIMIT) heap.push_back(make_pair(child_node, parent_node));
        if (maxcachesize > NO_CACHE_LIMIT) {
            vector<Item> v;
            v.reserve(i+1);
            for (int j = 0; j <= i; ++j) {
                v.push_back(itemset[j]);
            }
            heap.push_back(make_tuple(child_node, parent_node, itemset[i], v));
        }
//        TrieLtdNode *node = &(heap.emplace_back(TrieLtdNode()));
//        TrieLtdNode *node = new TrieLtdNode();
        TrieLtdEdge newedge{itemset[i], child_node};
//        TrieLtdEdge newedge;
//        newedge.item = itemset[i];
//        newedge.subtrie = child_node;
        if (i == pos) parent_node->edges.insert(geqEdge_it, newedge);
        else { // new node added so add the edge without checking its place
//            parent_node->edges.reserve(1);
            parent_node->edges.push_back(newedge);
        }
        cachesize++;
        parent_node = child_node;
        sub_nodes--;
    }
    child_node->data = nodeDataManager->initData();
    return child_node;
}

void Cache_Ltd_Trie::decreaseItemset(vector<Item>& itemset){
    auto *cur_node = (TrieLtdNode *) root;
    vector<TrieLtdEdge>::iterator geqEdge_it;
    int i = 0;
    for (; i <  itemset.size() - 1; i++) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end()){ // item found
            cur_node = geqEdge_it->subtrie;
            cur_node->n_subnodes--;
        }
    }
    geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
    cur_node->edges.erase(geqEdge_it);
}

// insert itemset. Check from root and insert items only if they do not exist using addItemsetPart function
pair<Node *, bool> Cache_Ltd_Trie::insert(Array<Item> itemset, NodeDataManager *nodeDataManager) {
//    cout << "..............." << endl;
    auto *cur_node = (TrieLtdNode *) root;
    if (itemset.size == 0) {
        cachesize++;
        cur_node->data = nodeDataManager->initData();
        return {cur_node, true};
    }
//    cout << "itemset to insert: ";
//    printItemset(itemset, true);

    if (getCacheSize() >= maxcachesize && maxcachesize > 0) {
        cout << "cachesize before = " << getCacheSize() << endl;
//        wipe((TrieLtdNode *) root, wipe_type);
        wipe();
        cout << "cachesize after = " << getCacheSize() << endl;
        if (getCacheSize() >= maxcachesize) canwipe = false;
    }

    vector<TrieLtdEdge>::iterator geqEdge_it;
    vector<TrieLtdNode*> existing_nodes;
    forEach (i, itemset) {
//        cout << "item : " << itemset[i] << endl;
        for (const auto& edge: cur_node->edges) {
//            cout << "edge : " << edge.item << endl;
        }
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != itemset[i]) { // the item does not exist
            int sub_nodes = itemset.size -  i;
            for (const auto node:existing_nodes) {
                node->n_subnodes += sub_nodes;
            }
            // create path representing the part of the itemset not yet present in the trie.
            TrieLtdNode *last_inserted_node = addNonExistingItemsetPart(itemset, i, geqEdge_it, cur_node, nodeDataManager);
            return {last_inserted_node, true};
        } else {
            existing_nodes.push_back(geqEdge_it->subtrie);
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

//void Cache_Ltd_Trie::wipe(Node *node1, float red_factor, WipeType wipe_type, Depth depth) {
void Cache_Ltd_Trie::wipe(float red_factor) {
    int n_del = (int) (maxcachesize * red_factor);
    cout << "n_del: " << n_del << endl;

    /*make_heap(heap.begin(), heap.end(), minHeapOrder);
    for (int i = 0; i < n_del; ++i) {
        if (heap.front().count_opti_path > 0) break;
        pop_heap(heap.begin(), heap.end(), minHeapOrder);
        heap.back().invalidateChildren();
        heap.pop_back();
    }*/

    /*if( heap[0] == nullptr) cout << "first is null" << endl;
    else {
        cout << "first not null" << endl;
        cout << heap.at(0)->count_opti_path << endl;
    }*/
//    for (const auto n : heap) if (n == nullptr) cout << "n null" << endl;
//    cout << "aucun null" << endl;
    sort(heap.begin(), heap.end(), sortDecOrder);
//    for (const auto &n : heap) if (n == nullptr) cout << "n null" << endl;
//    cout << "aucun null" << endl;
//    for(auto it = heap.rbegin(); it != heap.rend(); it++) if (*it == nullptr) cout << "it null" << endl;
//    cout << "aucun it null" << endl;
//cout << endl << "subnodes" << endl;
//    for (auto it = heap.rbegin(); it != heap.rend(); it++) if (std::get<0>(*it)->count_opti_path < 0) cout << std::get<0>(*it)->count_opti_path << ",";
//    for (auto it = heap.rbegin(); it != heap.rend(); it++) if (std::get<0>(*it) != nullptr) cout << std::get<0>(*it)->n_subnodes << ",";
//    cout << endl << endl;
    int counter = 0;
    for (auto it = heap.rbegin(); it != heap.rend(); it++) {
//    for (auto it = heap.rbegin(); it != heap.rend(); it++) {
//        if (it - heap.rbegin() == n_del || (*it != nullptr && (*it)->count_opti_path > 0)) break;
//        if ((*it != nullptr && (*it)->count_opti_path > 0)) break;
//cout << "n_iter : " << it - heap.rbegin() << endl;
//        if (counter == n_del || (*it).first->count_opti_path > 0) {
        if (counter == n_del || std::get<0>(*it)->count_opti_path > 0) {
//            cout << (*it)->count_opti_path << endl;
//            cout << "break" << endl;
            break;
        }
//        heap.back().first->invalidateChildren();
        //std::get<0>(heap.back())->invalidateChildren();
//        vector<pair<TrieLtdNode*,vector<TrieLtdEdge>*>>* hp = &heap;
//        auto is_target = [hp](TrieLtdEdge* look){ return look->subtrie == hp->back().first; };
//        auto it_node = find_if(heap.back().second->begin(),heap.back().second->end(), is_target);

//cout << "\nnode we look for: " << std::get<0>(heap.back()) << " its parent " << std::get<1>(heap.back()) << "its effort: "  <<  std::get<1>(heap.back())->solution_effort << " its subnodes: " << std::get<0>(heap.back())->n_subnodes << " its count opti: " << std::get<0>(heap.back())->count_opti_path << endl;
//cout << "item to look for: " << std::get<2>(heap.back()) << endl;
//cout << "itemset: ";
//        for (auto i:std::get<3>(heap.back())) {
//            cout << i << ",";
//        }
//        cout << endl;
//        cout << "items: ";
//        for (auto i:std::get<1>(heap.back())->edges) {
//            cout << i.item << ",";
//        }
//        cout << endl;
//        auto it_node = heap.back().second->edges.begin();
//        auto it_node = std::get<1>(heap.back())->edges.begin();
//        for(;it_node != heap.back().second->edges.end(); it_node++){
//        for(;it_node != std::get<1>(heap.back())->edges.end(); it_node++){
////            if (it_node->subtrie == heap.back().first) {
//            /*if (it_node->subtrie == std::get<0>(heap.back())) {
//                cout << "yes" << endl;
//                break;
//            }*/
//            if (it_node->item == std::get<2>(heap.back())) {
////                cout << "item found!!!" << endl;
//                if (it_node->subtrie == std::get<0>(heap.back())) {
////                    cout << "child found" << endl;
//                    break;
//                }
////                else cout << "child not found" << endl;
//            }
//        }
//        if (it_node->subtrie != heap.back().first) cout << "pas yes" << endl;
//        if (it_node->subtrie != std::get<0>(heap.back())) cout << "pas yes" << endl;
//        else cout << "yes" << endl;
//        delete it_node->subtrie;
//        delete heap.back().first;
        delete std::get<0>(heap.back());
//        heap.back().second->edges.erase(it_node);
//        std::get<1>(heap.back())->edges.erase(it_node);
        decreaseItemset(std::get<3>(heap.back()));
        heap.pop_back();
//        delete *it;
//        heap.erase(it);
        counter++;
    }
}

void Cache_Ltd_Trie::printload(TrieLtdNode *node, vector<Item> &itemset) {
    for (auto & edge : node->edges) {
        itemset.push_back(edge.item);
        printload(edge.subtrie, itemset);
        itemset.pop_back();
    }
}

void Cache_Ltd_Trie::updateItemsetLoad(Array<Item> itemset, bool inc) {
    auto *cur_node = (TrieLtdNode *) root;
    vector<TrieLtdEdge>::iterator geqEdge_it;
    forEach (i, itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end() && geqEdge_it->item == itemset[i]) { // item found
            cur_node = geqEdge_it->subtrie;
            if (inc) cur_node->count_opti_path++;
            else cur_node->count_opti_path--;
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
