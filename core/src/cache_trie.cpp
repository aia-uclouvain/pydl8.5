#include "cache_trie.h"
#include "rCoverWeight.h"
#include "rCoverFreq.h"
#include <algorithm>
#include "logger.h"
#include <cstdlib>

using namespace std;

bool lessTrieEdge(const TrieEdge edge, const Item item) {
    return edge.item < item;
}

bool lessEdgeItem(const TrieEdge edge1, const TrieEdge edge2) {
    return edge1.item < edge2.item;
}

// sort the nodes in decreasing order in the aim to get the lowest in O(1) as it will be the last
// ascending order ===> return true if a < b and false if a > b
// descending order ===> return true if a > b and false if a < b
//bool sortDecOrder(const pair<TrieLtdNode*,vector<Item>> &pair1, const pair<TrieLtdNode*,vector<Item>> &pair2) {
bool sortDecOrder(const pair<pair<TrieNode*,Itemset>, pair<TrieNode*,Itemset>> &pair1, const pair<pair<TrieNode*,Itemset>, pair<TrieNode*,Itemset>> &pair2) {
    const auto node1 = pair1.first.first, node2 = pair2.first.first;
    if (node1->count_opti_path > 0 && node2->count_opti_path == 0) return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->count_opti_path == 0 && node2->count_opti_path > 0) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}

bool sortReuseDecOrder(const pair<TrieNode*,TrieNode*> &pair1, const pair<TrieNode*,TrieNode*> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->count_opti_path > 0 && node2->count_opti_path == 0) return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->count_opti_path == 0 && node2->count_opti_path > 0) return false; // same for the node2
    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth) return node1->depth < node2->depth; // depth from bigger to lower, so sorted in creasing order
//    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth) return node1->support > node2->support;
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}

// in case of heap decreasing comparator for sort provides a min-heap
//bool minHeapOrder(const pair<TrieLtdNode*,vector<Item>> &pair1, const pair<TrieLtdNode*,vector<Item>> &pair2) {
/*bool minHeapOrder(const pair<TrieNode*,TrieNode*> &pair1, const pair<TrieNode*,TrieNode*> &pair2) {
    return sortDecOrder(pair1, pair2);
}*/

Cache_Trie::Cache_Trie(Depth maxdepth, WipeType wipe_type, int maxcachesize, float wipe_factor) : Cache(maxdepth, wipe_type, maxcachesize), wipe_factor(wipe_factor) {
    root = new TrieNode;
    if (maxcachesize > NO_CACHE_LIMIT) heap.reserve(maxcachesize - 1);
}

// look for itemset in the trie from root. Return null if not exist and the node of the last item if it exists
Node *Cache_Trie::get(const Itemset &itemset) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    for (const auto &item : itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != item) return nullptr; // item not found so itemset not found
        else cur_node = geqEdge_it->subtrie;
    }
    return cur_node;
}

int Cache_Trie::getCacheSize() {
    if (maxcachesize == NO_CACHE_LIMIT) return cachesize;
    else return heap.size() + 1;
}

// classic top down
TrieNode *Cache_Trie::addNonExistingItemsetPart(Itemset &itemset, int pos, vector<TrieEdge>::iterator &geqEdge_it, TrieNode *parent_node) {
    TrieNode* child_node;
    for (int i = pos; i < itemset.size(); ++i) {
//        if (itemset.size() - pos > 1) cout << "PRAPRAPRAPRPAPRPAPRAPRPAPRPAPRPAPRPAPRPAPRPAP" << endl;
        child_node = new TrieNode();
        Itemset child_itst, parent_itst;
        for (int j = 0; j < i; ++j) {
            child_itst.push_back(itemset[j]);
            parent_itst.push_back(itemset[j]);
        }
        child_itst.push_back(itemset[i]);
//        if (parent_itst.size() == 1 and parent_itst.at(0) == 10 and child_itst.size() == 2 and child_itst.at(0) == 10 and child_itst.at(1) == 16){
//        if (child_itst.size() == 2 and child_itst.at(0) == 10 and child_itst.at(1) == 16){
//            cout << "ins child " << child_node << " ";
//            printItemset(child_itst, true, true);
//            cout << "ins par " << parent_node << " ";
//            printItemset(parent_itst, true, true);
//            cout << "parent children (" << parent_node->edges.size() << ") ==>";
//            for (const auto e: parent_node->edges) cout << e.item << ":" << e.subtrie << ",";
//            cout << endl;
//        }
        if (maxcachesize > NO_CACHE_LIMIT) {
            heap.push_back(make_pair(make_pair(child_node,child_itst), make_pair(parent_node, parent_itst)));
            child_node->trie_parent = parent_itst;
        }
        TrieEdge newedge{itemset[i], child_node};
//        if (itemset.size() == 1 and itemset.at(0) == 10){
//            cout << " pos:" << i << " item:" << itemset[i] << " child@:" << child_node << endl;
//        }
        if (i == pos) parent_node->edges.insert(geqEdge_it, newedge);
        else parent_node->edges.push_back(newedge); // new node added so add the edge without checking its place
//        if (itemset.size() == 1 and itemset.at(0) == 10){
//            for (const auto e: ((TrieNode*)root)->edges){
//                if (e.item == 10){
//                    cout << "child root" << e.subtrie;
//                }
//            }
//            for (const auto e: parent_node->edges){
//                if (e.item == 10){
//                    cout << "child parent" << e.subtrie;
//                }
//            }
//        }
        cachesize++;
        child_node->depth = i + 1;
        parent_node = child_node;
        for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
        Logger::showMessage(":(", child_node->count_opti_path, ") -- ");
    }
    return child_node;
}

// insert itemset. Check from root and insert items only if they do not exist using addItemsetPart function
pair<Node*, bool> Cache_Trie::insert(Itemset &itemset) {
    auto *cur_node = (TrieNode *) root;
    if (itemset.empty()) {
        cachesize++;
        return {cur_node, true};
    }



    /*auto ii = find(itemset.begin(), itemset.end(), 10);
    if (ii != itemset.end()){
        ++ii;
        if (*ii == 16) {
            cout << "\nins par trou ";
            printItemset(itemset, true, true);
        }
    }*/

//    if (itemset.size() >= 2 and itemset.at(0) == 10 and itemset.at(1) == 16){
//        cout << "\nins par trou ";
//        printItemset(itemset, true, true);
//    }

//    if (getCacheSize() >= maxcachesize && maxcachesize > NO_CACHE_LIMIT) {
//        Logger::showMessageAndReturn("wipe launched");
////        cout << "wipe" << endl;
//        wipe();
////        cout << "done" << endl;
//        Logger::showMessageAndReturn("wipe done");
//    }

    Logger::showMessage("increasing load of itemset: ");
    printItemset(itemset);

    vector<TrieEdge>::iterator geqEdge_it;
    for (int i = 0; i < itemset.size(); ++i) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != itemset[i]) { // the item does not exist

            if (getCacheSize() + itemset.size() - i > maxcachesize && maxcachesize > NO_CACHE_LIMIT) {
                Logger::showMessageAndReturn("wipe launched");
                cout << "wipe" << endl;
                wipe();
                geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
//        cout << "done" << endl;
                Logger::showMessageAndReturn("wipe done");
            }

            // create path representing the part of the itemset not yet present in the trie.
            TrieNode *last_inserted_node = addNonExistingItemsetPart(itemset, i, geqEdge_it, cur_node);
            Logger::showMessageAndReturn("");

            /*if ( not isLoadConsistent((TrieNode*)root, itemset) ) {
                cout << "insert load_const: 0" << endl;
                exit(0);
            }*/

            if (itemset.size() >= 3 and itemset.at(0) == 5 and itemset.at(1) == 8 and itemset.at(2) == 14){
                Itemset t;
                t.push_back(5);
                t.push_back(8);
                t.push_back(14);
                auto* n = get(t);
                cout << "create 5,8,14:" << n->count_opti_path << endl;
                printItemset(itemset, true);
            }
            if (itemset.size() >= 2 and itemset.at(0) == 5 and itemset.at(1) == 14){
                Itemset t;
                t.push_back(5);
                t.push_back(14);
                auto* n = get(t);
                cout << "create 5,14:" << n->count_opti_path << endl;
                printItemset(itemset, true);
            }

            return {last_inserted_node, true};
        } else {
            if (i == 0) cur_node->n_reuse++; // root node
            cur_node = geqEdge_it->subtrie;
            cur_node->count_opti_path++;
            cur_node->n_reuse++;
            cur_node->depth = i + 1;
            for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
            Logger::showMessage(":(", cur_node->count_opti_path, ") -- ");
        }
    }
    Logger::showMessageAndReturn("");

    /*if ( not isLoadConsistent((TrieNode*)root, itemset) ) {
        cout << "insert load_const: 0" << endl;
        exit(0);
    }*/

    if (itemset.size() >= 3 and itemset.at(0) == 5 and itemset.at(1) == 8 and itemset.at(2) == 14){
        Itemset t;
        t.push_back(5);
        t.push_back(8);
        t.push_back(14);
        auto* n = get(t);
        cout << "create existing 5,8,14:" << n->count_opti_path << endl;
        printItemset(itemset, true);
    }
    if (itemset.size() >= 2 and itemset.at(0) == 5 and itemset.at(1) == 14){
        Itemset t;
        t.push_back(5);
        t.push_back(14);
        auto* n = get(t);
//        cout << "create existing 5,14:" << n->count_opti_path << " " << ((FND)n->data)->test << endl;
        printItemset(itemset, true);
    }

    if (cur_node->data == nullptr) return {cur_node, true};
    else return {cur_node, false};
}

void wipeAll(TrieNode *node) {
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

void Cache_Trie::wipe() {
    int n_del = (int) (maxcachesize * wipe_factor);

    /*if (wipe_type == All){ // the problem here is that the priority queue still contain the removed nodes
        wipeAll((TrieLtdNode*)root);
        return;
    }*/

    switch (wipe_type) {
        case Subnodes:
            computeSubNodes((TrieNode*)root);
//             cout << "is subnodes hierarchy consistent : " << isConsistent((TrieNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            break;
        case Recall:
            // cout << "is reuse hierarchy consistent : " << isConsistent((TrieLtdNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
//            sort(heap.begin(), heap.end(), sortReuseDecOrder);
            break;
        default: // All. this block is used when the above if is commented.
            computeSubNodes((TrieNode*)root);
            // cout << "is subnodes hierarchy consistent for all wipe : " << isConsistent((TrieLtdNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            n_del = heap.size();
    }

//     cout << "cachesize before wipe = " << getCacheSize() << endl;
    int counter = 0;
    for (auto it = heap.rbegin(); it != heap.rend(); it++) {
        if (counter == n_del || it->first.first->count_opti_path > 0) break;

        bool test = false;
        // remove the edge bringing to the node
        if (it->first.second.size() == 3 and it->first.second.at(0) == 5 and it->first.second.at(1) == 8 and it->first.second.at(2) == 14){
            cout << "delo " << it->first.first->count_opti_path << endl;
            test = true;
        }
//        int aa = rand();

//        if (it->first.second.size() == 1 and it->first.second.at(0) == 10){
//            cout << "\npar del" << endl;
//            cout << "par:" << it->first.first << " load:" << it->first.first->count_opti_path << " subn:" << it->first.first->n_subnodes << " depth:" << it->first.first->depth << endl;
//            Itemset t;
//            t.push_back(10);
//            t.push_back(16);
//            TrieNode* n = (TrieNode*)get(t);
//            if (n) cout << "chil:" << n << " load:" << n->count_opti_path << " subn:" << n->n_subnodes << " depth:" << n->depth << endl;
//cout << "aa:" << aa << endl;
//            cout << "par child (" << it->first.first->edges.size() << ") ==>";
//            for (const auto e: it->first.first->edges) cout << e.item << ":" << e.subtrie << ",";
//            cout << endl;
//            if (n){
//                cout << "child child (" << n->edges.size() << ") ==>";
//                for (const auto e: n->edges) cout << e.item << ":" << e.subtrie << ",";
//                cout << endl;
//                cout << "subconst:" << isConsistent((TrieNode*)root) << endl;
//            }
//        }
//        if (it->first.second.size() == 2 and it->first.second.at(0) == 10 and it->first.second.at(1) == 16){
//            cout << "\nbvbv child_itemset : ";
//            printItemset(it->first.second, true, true);
//            cout << "parent_itemset : ";
//            printItemset(it->second.second, true, true);
//            cout << "child:" << it->first.first << " load:" << it->first.first->count_opti_path << " subn:" << it->first.first->n_subnodes << " depth:" << it->first.first->depth << endl;
//            cout << "parent:" << it->second.first << " load:" << it->second.first->count_opti_path << " subn:" << it->second.first->n_subnodes << " depth:" << it->second.first->depth << endl;
////            Itemset t;
////            t.push_back(10);
////            TrieNode* n = (TrieNode*)get(t);
////            if (n){
////                for(const auto edge: n->edges) cout << edge.item << ":" << edge.subtrie << ", ";
////                cout << n << endl;
////            }
////            else cout << "10 not exist" << endl;
////            for(const auto edge: it->second.first->edges) cout << edge.item << ":" << edge.subtrie << ", ";
////            cout << it->second.first->edges.size() << endl;
//            cout << "parent children (" << it->second.first->edges.size() << ") ==>";
//            for (const auto e: it->second.first->edges) cout << e.item << ":" << e.subtrie << ",";
//            cout << endl;
//            cout << "aa:" << aa << endl;
//
//        }

//        auto child_edge_it = find_if(it->second.first->edges.begin(), it->second.first->edges.end(), [it](const TrieEdge &look) { return look.subtrie == it->first.first; });
//        Attribute child_attr = item_attribute(child_edge_it->item);
//        if (test) cout << "attr found: " << child_attr << endl;

        auto child_edge_it = find_if(((TrieNode*)get(it->first.first->trie_parent))->edges.begin(), ((TrieNode*)get(it->first.first->trie_parent))->edges.end(), [it](const TrieEdge &look) { return look.subtrie == it->first.first; });





        for (const auto &search_parent_itemset : it->first.first->search_parents) {
            auto search_parent_node = get(search_parent_itemset);
            if ( search_parent_node != nullptr and search_parent_node->data != nullptr ) {

                Itemset child_itemset_left = addItem(search_parent_itemset, item(((FND)search_parent_node->data)->test, NEG_ITEM));
                Itemset child_itemset_right = addItem(search_parent_itemset, item(((FND)search_parent_node->data)->test, POS_ITEM));
                Node* child_itemset_left_node = get(child_itemset_left);
                Node* child_itemset_right_node = get(child_itemset_right);

                if ( (child_itemset_left_node != nullptr and child_itemset_left_node == it->first.first) or (child_itemset_right_node != nullptr and child_itemset_right_node == it->first.first) ) {
                    ((FND)search_parent_node->data)->test = -((FND)search_parent_node->data)->test;
                    ((FND)search_parent_node->data)->left = nullptr;
                    ((FND)search_parent_node->data)->right = nullptr;
                    if ( ((FND)search_parent_node->data)->error < FLT_MAX ) {
                        ((FND)search_parent_node->data)->lowerBound = ((FND)search_parent_node->data)->error;
                        ((FND)search_parent_node->data)->error = FLT_MAX;
                    }
                }

            }
        }

        if (it->first.second.size() == 3 and it->first.second.at(0) == 5 and it->first.second.at(1) == 8 and it->first.second.at(2) == 14){
            cout << "DELLLAAAO" << endl;
        }

        auto trie_parent_itemset = it->first.first->trie_parent;
        auto trie_parent_node = get(trie_parent_itemset);

        if (it->first.second.size() == 3 and it->first.second.at(0) == 5 and it->first.second.at(1) == 8 and it->first.second.at(2) == 14){
            printItemset(trie_parent_itemset, true);
            cout << "par_node " << trie_parent_node << endl;
        }


        if ( trie_parent_node != nullptr and trie_parent_node->data != nullptr ) {

            Itemset child_itemset_left = addItem(trie_parent_itemset, item(((FND)trie_parent_node->data)->test, NEG_ITEM));
            Itemset child_itemset_right = addItem(trie_parent_itemset, item(((FND)trie_parent_node->data)->test, POS_ITEM));
            Node* child_itemset_left_node = get(child_itemset_left);
            Node* child_itemset_right_node = get(child_itemset_right);

            if (it->first.second.size() == 3 and it->first.second.at(0) == 5 and it->first.second.at(1) == 8 and it->first.second.at(2) == 14){
                cout << "child left itst ";
                printItemset(child_itemset_left, true);
                cout << "child right itst ";
                printItemset(child_itemset_right, true);
            }

            if (it->first.second.size() == 3 and it->first.second.at(0) == 5 and it->first.second.at(1) == 8 and it->first.second.at(2) == 14){
                cout << "self " << it->first.first << endl;
                cout << "par_left_child_node " << child_itemset_left_node << endl;
                cout << "par_right_child_node " << child_itemset_right_node << endl;
            }

            if ( (child_itemset_left_node != nullptr and child_itemset_left_node == it->first.first) or (child_itemset_right_node != nullptr and child_itemset_right_node == it->first.first) ) {
                ((FND)trie_parent_node->data)->test = -((FND)trie_parent_node->data)->test;
                ((FND)trie_parent_node->data)->left = nullptr;
                ((FND)trie_parent_node->data)->right = nullptr;
                if ( ((FND)trie_parent_node->data)->error < FLT_MAX ) {
                    ((FND)trie_parent_node->data)->lowerBound = ((FND)trie_parent_node->data)->error;
                    ((FND)trie_parent_node->data)->error = FLT_MAX;
                }
            }

        }


        it->second.first->edges.erase(child_edge_it);






//        for (const auto& parento : it->first.first->search_parents) {
//            auto parent = get(parento);
//            if ( parent != nullptr and parent->data != nullptr and ((FND)parent->data)->test == child_attr ) {
//                ((FND)parent->data)->test = -1;
//                ((FND)parent->data)->left = nullptr;
//                ((FND)parent->data)->right = nullptr;
//                if ( ((FND)parent->data)->error < FLT_MAX ) {
//                    ((FND)parent->data)->lowerBound = ((FND)parent->data)->error;
//                    ((FND)parent->data)->error = FLT_MAX;
//                }
//            }
//        }
//        auto trie_parent = get(it->first.first->trie_parent);
//        if ( trie_parent != nullptr and trie_parent->data != nullptr and ((FND)trie_parent->data)->test == child_attr ) {
//            ((FND)trie_parent->data)->test = -1;
//            ((FND)trie_parent->data)->left = nullptr;
//            ((FND)trie_parent->data)->right = nullptr;
//            if ( ((FND)trie_parent->data)->error < FLT_MAX ) {
//                ((FND)trie_parent->data)->lowerBound = ((FND)trie_parent->data)->error;
//                ((FND)trie_parent->data)->error = FLT_MAX;
//            }
//        }







//        else if (test) cout << "cond non bon " << ((FND)trie_parent->data)->test << endl;
//        if (test) {
//            Itemset t;
//            t.push_back(2);
//            t.push_back(14);
//            auto* n = get(t);
//            if (n) cout << "test " << ((FND)n->data)->test << endl;
//        }
        // remove the node
        delete it->first.first;
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

int Cache_Trie::computeSubNodes(TrieNode* node) {
    node->n_subnodes = 0;
    if (node->edges.empty()) { return 0; }
    for (auto& edge: node->edges) node->n_subnodes += 1 + computeSubNodes(edge.subtrie);
    return node->n_subnodes;
}


/*bool Cache_Trie::isLoadConsistent(TrieNode* node){
    return all_of(node->edges.begin(), node->edges.end(), [this] (const TrieEdge &edge) {
        TrieNode *child = edge.subtrie;
        auto loadSum = [child]() {
            int sum = 0;
            for (const auto& edge : child->edges) sum += edge.subtrie->count_opti_path;
            return sum + 1;
        };
        if (child->count_opti_path != loadSum() or not isLoadConsistent(child)) return false;
        return true;
    });
}*/

bool Cache_Trie::isLoadConsistent(TrieNode* node, Itemset itemset){
    TrieNode *parent = node;
    if (parent->edges.empty()) return true;
    auto loadSum = [parent]() {
        int sum = 1;
        for (const auto& edge : parent->edges) sum += edge.subtrie->count_opti_path;
        return sum;
    };
    if (parent != root and not(parent->count_opti_path == 0 and loadSum() == 1) and parent->count_opti_path < loadSum()) {
        cout << "par_ite:";
        printItemset(itemset, true, false);
        cout << " par:" << parent->count_opti_path << " load_sum:" << loadSum() << endl;
        return false;
    }
    return all_of(parent->edges.begin(), parent->edges.end(), [itemset, this] (const TrieEdge &edge) {
        TrieNode *child = edge.subtrie;
        Itemset itemset1 = itemset;
        itemset1.push_back(edge.item);
        if (not isLoadConsistent(child, itemset1)) return false;
        return true;
    });

    /*for (const auto& edge: node->edges) {
        TrieNode *child = edge.subtrie;
        if (not isConsistent(child)) return false;
    }
    return true;*/
}

bool Cache_Trie::isNonNegConsistent(TrieNode* node){
    return all_of(node->edges.begin(), node->edges.end(), [this] (const TrieEdge &edge) {
        TrieNode *child = edge.subtrie;
        if (child->count_opti_path < 0 or not isNonNegConsistent(child)) return false;
        return true;
    });
}



bool Cache_Trie::isConsistent(TrieNode* node){
    return all_of(node->edges.begin(), node->edges.end(), [node,this] (const TrieEdge &edge) {
        TrieNode *parent = node, *child = edge.subtrie;
        if(
                ((wipe_type == Subnodes || wipe_type == All) && child->n_subnodes >= parent->n_subnodes) or
                (wipe_type == Recall && (child->n_reuse > parent->n_reuse || (child->n_reuse == parent->n_reuse && child->depth <= parent->depth))) or
                (not isConsistent(child))
                ) return false;
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

void Cache_Trie::updateItemsetLoad(Itemset &itemset, bool inc) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
//    if (inc) Logger::showMessage("increasing load of itemset: ");
//    else Logger::showMessage("decreasing load of itemset: ");
//    printItemset(itemset);
//    if (inc) cout << "increasing load of itemset: ";
//    else cout << "decreasing load of itemset: ";
//    printItemset(itemset, true, true);
    int i = 0;
    for (const auto item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end() && geqEdge_it->item == item) { // item found
            cur_node = geqEdge_it->subtrie;
            if (inc) cur_node->count_opti_path++;
            else cur_node->count_opti_path--;

            Itemset its;
            for (int j = 0; j <= i; ++j) its.push_back(itemset.at(j));
            if (its.size() == 3 and its.at(0) == 5 and its.at(1) == 8 and its.at(2) == 14){
                cout << "incr 5,8,14: " << cur_node->count_opti_path << endl;
                if (cur_node->count_opti_path == 0) {
                    Itemset t;
                    t.push_back(5);
                    t.push_back(8);
                    auto* n = get(t);
//                    if (not n){
//                        t.clear();
//                        t.push_back(0);
//                        t.push_back(2);
//                        auto* n = get(t);
//                        cout << "0, 2, test " << ((FND)n->data)->test << endl;
//                    }
                    if (n and n->data) cout << "5, 8 test " << ((FND)n->data)->test << endl;
                }
            }
            if (its.size() == 2 and its.at(0) == 5 and its.at(1) == 14){
                cout << "incr 5,14: " << cur_node->count_opti_path << endl;
                if (cur_node->count_opti_path == 0) {
                    Itemset t;
                    t.push_back(5);
                    auto* n = get(t);
                    if (n and n->data) cout << "5 test " << ((FND)n->data)->test << endl;
                }
            }

//            for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
//            Logger::showMessage(":(", cur_node->count_opti_path, ") -- ");

//            for (int j = 0; j <= i; ++j) cout << itemset.at(j) << ",";
//            cout << ":(" << cur_node->count_opti_path << ") -- ";

//            printItemset(itemset, false, false);
//            Logger::showMessageAndReturn(" load: ", cur_node->count_opti_path);
            if (cur_node->count_opti_path < 0) {
                cout << "load itemset: ";
                for (int j = 0; j <= i; ++j) cout << itemset.at(j) << ",";
                cout << " " << cur_node->count_opti_path << endl;
//                exit(0);
            }
//            if (cur_node->count_opti_path < 0) cout << "load = " << cur_node->count_opti_path << endl;
        }
        else {
            cout << "probleme" << endl;

            Itemset its;
            for (int j = 0; j <= i; ++j) its.push_back(itemset.at(j));
            printItemset(its, true);

            Itemset t;
            t.push_back(0);
            t.push_back(2);
            auto* n = get(t);
//            cout << "test " << ((FND)n->data)->test << endl;
        }
        i++;
    }
    Logger::showMessageAndReturn("");
//    cout << endl;
}

void Cache_Trie::updateSubTreeLoad(Itemset &itemset, Item firstItem, Item secondItem, bool inc) {
    for (auto item: {firstItem, secondItem}) {
        if (item == -1) {
            if (item == secondItem) Itemset().swap(itemset);
            continue;
        }

        Itemset child_itemset = addItem(itemset, item);

        // when we build both branches itemsets, we don't need the parent anymore
        if (item == secondItem) Itemset().swap(itemset);

        updateItemsetLoad(child_itemset, inc);
        auto *child_node = (TrieNode *) get(child_itemset);
//        Itemset().swap(child_itemset);

//        if (child_node and child_node->data and ((FND) child_node->data)->error < FLT_MAX and ((FND) child_node->data)->left and ((FND) child_node->data)->right) {
        if (((FND) child_node->data)->left and ((FND) child_node->data)->right) {
            Item nextFirstItem = item(((FND) child_node->data)->test, NEG_ITEM);
            Item nextSecondItem = item(((FND) child_node->data)->test, POS_ITEM);
            updateSubTreeLoad(child_itemset, nextFirstItem, nextSecondItem, inc);
        }
        Itemset().swap(child_itemset);
    }
}


void Cache_Trie::printItemsetLoad(Itemset &itemset, bool inc) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    cout << "printing load of itemset: ";
    printItemset(itemset, true, true);
    int i = 0;
    for (const auto& item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end() && geqEdge_it->item == item) { // item found
            cur_node = geqEdge_it->subtrie;
//            for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
//            Logger::showMessage(":(", cur_node->count_opti_path, ") -- ");

            for (int j = 0; j <= i; ++j) cout << itemset.at(j) << ",";
            cout << ":(" << cur_node->count_opti_path << ") -- ";

//            printItemset(itemset, false, false);
//            Logger::showMessageAndReturn(" load: ", cur_node->count_opti_path);
        }
        i++;
    }
    cout << endl;
}


void Cache_Trie::printSubTreeLoad(Itemset &itemset, Item firstItem, Item secondItem, bool inc) {
    for (auto item: {firstItem, secondItem}) {

        if (item == -1) { if (item == secondItem) Itemset().swap(itemset); continue; }

        Itemset child_itemset = addItem(itemset, item);
        if (item == secondItem) Itemset().swap(itemset); // when we build both branches itemsets, we don't need the parent anymore
        printItemsetLoad(child_itemset, inc);

        auto *child_node = (TrieNode *) get(child_itemset);
        if (child_node and child_node->data and ((FND) child_node->data)->left and ((FND) child_node->data)->right) {
            Item nextFirstItem = item(((FND) child_node->data)->test, NEG_ITEM);
            Item nextSecondItem = item(((FND) child_node->data)->test, POS_ITEM);
            printSubTreeLoad(child_itemset, nextFirstItem, nextSecondItem, inc);
        }
        Itemset().swap(child_itemset);
    }
}

void Cache_Trie::updateRootPath(Itemset &itemset, int value) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    for (const auto& item : itemset) {
        cur_node = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge)->subtrie;
        cur_node->count_opti_path += value;
    }
}

//void Cache_Trie::updateParents(Node *best, Node *left, Node *right) {
//    auto *best_ = (TrieNode*) best, *left_ = (TrieNode*) left, *right_ = (TrieNode*) right;
//    left_->search_parents.push_back(best_);
//    right_->search_parents.push_back(best_);
//}































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
