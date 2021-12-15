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
/*bool sortDecOrder( TrieNode *&pair1,  TrieNode *&pair2) {
    const auto node1 = pair1, node2 = pair2;
//    TrieNode* node1 = (*(pair1));
//    TrieNode* node2 = (*(pair2));
//cout << "loads: " << node1->count_opti_path << " " << node2->count_opti_path << endl;
    if (node1->count_opti_path > 0 && node2->count_opti_path <= 0)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->count_opti_path <= 0 && node2->count_opti_path > 0) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}*/
/*bool sortDecOrder( TrieNode *&pair1,  TrieNode *&pair2) {
    const auto node1 = pair1, node2 = pair2;
    if (node1->is_used and not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used and node2->is_used) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}*/

bool sortDecOrder( pair<TrieNode*,Itemset> &pair1, pair<TrieNode*,Itemset> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->is_used and not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used and node2->is_used) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}

/*bool sortReuseDecOrder(TrieNode *&pair1, TrieNode *&pair2) {
    const auto node1 = pair1, node2 = pair2;
    if (node1->count_opti_path > 0 && node2->count_opti_path == 0)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->count_opti_path == 0 && node2->count_opti_path > 0) return false; // same for the node2
    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth)
        return node1->depth < node2->depth; // depth from bigger to lower, so sorted in creasing order
//    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth) return node1->support > node2->support;
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}*/

bool sortReuseDecOrder(pair<TrieNode*,Itemset> &pair1, pair<TrieNode*,Itemset> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->count_opti_path > 0 && node2->count_opti_path == 0)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->count_opti_path == 0 && node2->count_opti_path > 0) return false; // same for the node2
    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth)
        return node1->depth < node2->depth; // depth from bigger to lower, so sorted in creasing order
//    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth) return node1->support > node2->support;
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}

// in case of heap decreasing comparator for sort provides a min-heap
//bool minHeapOrder(const pair<TrieLtdNode*,vector<Item>> &pair1, const pair<TrieLtdNode*,vector<Item>> &pair2) {
/*bool minHeapOrder( TrieNode *&pair1,  TrieNode *&pair2) {
    return sortDecOrder(pair1, pair2);
}*/

bool minHeapOrder( pair<TrieNode*,Itemset> &pair1, pair<TrieNode*,Itemset> &pair2) {
    return sortDecOrder(pair1, pair2);
}

Cache_Trie::Cache_Trie(Depth maxdepth, WipeType wipe_type, int maxcachesize, float wipe_factor) : Cache(maxdepth,
                                                                                                        wipe_type,
                                                                                                        maxcachesize),
                                                                                                  wipe_factor(
                                                                                                          wipe_factor) {
    root = new TrieNode;
    if (maxcachesize > NO_CACHE_LIMIT) heap.reserve(maxcachesize - 1);
}

// look for itemset in the trie from root. Return null if not exist and the node of the last item if it exists
Node *Cache_Trie::get(const Itemset &itemset) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    for (const auto &item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != item)
            return nullptr; // item not found so itemset not found
        else cur_node = geqEdge_it->subtrie;
    }
    return cur_node;
}

Node *Cache_Trie::getandSet(const Itemset &itemset, int& n_used) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    for (const auto &item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != item)
            return nullptr; // item not found so itemset not found
        else {
            cur_node = geqEdge_it->subtrie;
            cur_node->is_used = true;
            ++n_used;
        }
    }
    return cur_node;
}

int Cache_Trie::getCacheSize() {
    if (maxcachesize == NO_CACHE_LIMIT) return cachesize;
    else return heap.size() + 1;
}

// classic top down
TrieNode *Cache_Trie::addNonExistingItemsetPart(Itemset &itemset, int pos, vector<TrieEdge>::iterator &geqEdge_it,
                                                TrieNode *parent_node) {
    TrieNode *child_node;
    for (int i = pos; i < itemset.size(); ++i) {
        child_node = new TrieNode();

        if (maxcachesize > NO_CACHE_LIMIT) {
//            heap.push_back(child_node);
            Itemset its;
            for (int j = 0; j <= i; ++j) {
                int e = itemset.at(j);
                its.push_back(e);
            }
            heap.push_back(make_pair(child_node, its));
//            cout << (*(heap.back()))->count_opti_path << endl;
//TrieNode* parent_node_cpy = parent_node;
            child_node->trie_parent = parent_node;
//            printItemset(itemset, true);
//            cout << "child @: " << child_node_cpy << endl;
//            cout << "@ in heap: " << (*(heap.back())) << endl;
//            cout << "wrapper @: " << child_node->self << endl;
//            cout << "wrapper @ in heap: " << heap.back() << endl;
//            cout << "first: " << (*(heap.at(0))) << endl;
        }

        TrieEdge newedge{itemset[i], child_node};
        if (i == pos) parent_node->edges.insert(geqEdge_it, newedge);
        else parent_node->edges.push_back(newedge); // new node added so add the edge without checking its place
        cachesize++;
        child_node->depth = i + 1;
        parent_node = child_node;
//        for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
//        Logger::showMessage(":(", child_node->count_opti_path, ") -- ");
    }
    return child_node;
}

// insert itemset. Check from root and insert items only if they do not exist using addItemsetPart function
pair<Node *, bool> Cache_Trie::insert(Itemset &itemset) {
    auto *cur_node = (TrieNode *) root;
    if (itemset.empty()) {
        cachesize++;
        return {cur_node, true};
    }

    Logger::showMessage("increasing load of itemset: ");
    printItemset(itemset);

    vector<TrieEdge>::iterator geqEdge_it;
    for (int i = 0; i < itemset.size(); ++i) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() or geqEdge_it->item != itemset[i]) { // the item does not exist

            if (getCacheSize() + itemset.size() - i > maxcachesize && maxcachesize > NO_CACHE_LIMIT) {
                Logger::showMessageAndReturn("wipe launched");
//                cout << "wipe" << endl;
                wipe();
                geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
//        cout << "done" << endl;
                Logger::showMessageAndReturn("wipe done");
            }

            // create path representing the part of the itemset not yet present in the trie.
            TrieNode *last_inserted_node = addNonExistingItemsetPart(itemset, i, geqEdge_it, cur_node);
            Logger::showMessageAndReturn("");


//            cout << "first_: " << endl;
//            cout << "first_: " << (*(heap.at(0))) << endl;

            return {last_inserted_node, true};
        } else {
            if (i == 0) cur_node->n_reuse++; // root node
            cur_node = geqEdge_it->subtrie;
//            cur_node->count_opti_path++;
            cur_node->n_reuse++;
            cur_node->depth = i + 1;
//            for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
//            Logger::showMessage(":(", cur_node->count_opti_path, ") -- ");
        }
    }
    Logger::showMessageAndReturn("");

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

/*void Cache_Trie::wipe() {
//    cout << "wipe " << getCacheSize() << endl;
//    cout << getCacheSize() << endl;

    int n_del = (int) (maxcachesize * wipe_factor);

//    if (wipe_type == All){ // the problem here is that the priority queue still contain the removed nodes
//        wipeAll((TrieLtdNode*)root);
//        return;
//    }
    int n_used = 0;
    setOptimalNodes((TrieNode*)root, n_used);
    Itemset itemset;
    setUsingNodes((TrieNode*)root, itemset, n_used);

    cout << "n_used: " << n_used << endl;

    switch (wipe_type) {
        case Subnodes:
            computeSubNodes((TrieNode *) root);
//             cout << "is subnodes hierarchy consistent : " << isConsistent((TrieNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            break;
        case Recall:
            // cout << "is reuse hierarchy consistent : " << isConsistent((TrieLtdNode*)root) << endl;
//            sort(heap.begin(), heap.end(), sortDecOrder);
            sort(heap.begin(), heap.end(), sortReuseDecOrder);
            break;
        default: // All. this block is used when the above if is commented.
            computeSubNodes((TrieNode *) root);
            // cout << "is subnodes hierarchy consistent for all wipe : " << isConsistent((TrieLtdNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            n_del = heap.size();
    }

//     cout << "cachesize before wipe = " << getCacheSize() << endl;
    int counter = 0;
    for (auto it = heap.rbegin(); it != heap.rend(); it++) {

        // the node to remove
        TrieNode *node_del = *it;

        // stop condition
        if (counter == n_del or node_del->is_used) break;

        // if the node to delete found its best solution, remove its pointer its left and right children
        if ( node_del->data != nullptr and node_del->data->left != nullptr ){
            TrieNode* node_del_left = ((TrieNode*)(node_del->data->left));
            TrieNode* node_del_right = ((TrieNode*)(node_del->data->right));
            node_del_left->search_parents.erase(std::find(node_del_left->search_parents.begin(), node_del_left->search_parents.end(), node_del));
            node_del_right->search_parents.erase(std::find(node_del_right->search_parents.begin(), node_del_right->search_parents.end(), node_del));
        }

        // remove from its search space parents, the fact that the node to delete is their best solution
        for (auto search_parent_node: node_del->search_parents) {

            cout << search_parent_node << "  ";

            if (search_parent_node != nullptr and search_parent_node->data != nullptr and ( search_parent_node->data->left == node_del or search_parent_node->data->right == node_del ) ) {

                // remove the fact that the parent found the best solution
                if ( search_parent_node->data->error < FLT_MAX) {
                    search_parent_node->data->lowerBound = search_parent_node->data->error; // set the error as lb to help the re-computation
                    search_parent_node->data->error = FLT_MAX;
                }
                search_parent_node->data->test = -search_parent_node->data->test; // keep the best attribute in order to explore it first during the re-computation

                // inform the corresponding left or right node (e.g. A will inform not A) that their parent won't recognize them anymore
                TrieNode* left_node = ((TrieNode*)search_parent_node->data->left);
                TrieNode* right_node = ((TrieNode*)search_parent_node->data->right);
                if ( search_parent_node->data->left == node_del )
                    right_node->search_parents.erase(std::find(right_node->search_parents.begin(), right_node->search_parents.end(), search_parent_node));
                if ( search_parent_node->data->right == node_del )
                    left_node->search_parents.erase(std::find(left_node->search_parents.begin(), left_node->search_parents.end(), search_parent_node));

                // invalidate the parent children
                search_parent_node->data->left = nullptr;
                search_parent_node->data->right = nullptr;

            }
        }

//        // remove from its trie parent the fact that the node to delete is its best solution
//        auto trie_parent_node = node_del->trie_parent;
//        if (trie_parent_node != nullptr and trie_parent_node->data != nullptr and
//            ( trie_parent_node->data->left == node_del or
//             trie_parent_node->data->right == node_del)) {
//
//            trie_parent_node->data->test = -trie_parent_node->data->test;
//            trie_parent_node->data->left = nullptr;
//            trie_parent_node->data->right = nullptr;
//            if (trie_parent_node->data->error < FLT_MAX) {
//                trie_parent_node->data->lowerBound = trie_parent_node->data->error;
//                trie_parent_node->data->error = FLT_MAX;
//            }
//
//        }

        // remove the edge bringing to the node
        node_del->trie_parent->edges.erase(find_if(node_del->trie_parent->edges.begin(), node_del->trie_parent->edges.end(), [node_del](const TrieEdge &look) { return look.subtrie == node_del; }));

        // remove the node
        delete node_del;
        heap.pop_back();
        counter++;
    }
    cout << endl;
//     cout << "cachesize after wipe = " << getCacheSize() << endl;
}*/

void Cache_Trie::wipe() {
//    cout << "wipe " << getCacheSize() << endl;
//    cout << getCacheSize() << endl;

    int n_del = (int) (maxcachesize * wipe_factor);

    /*if (wipe_type == All){ // the problem here is that the priority queue still contain the removed nodes
        wipeAll((TrieLtdNode*)root);
        return;
    }*/
    int n_used = 0;
    setOptimalNodes((TrieNode*)root, n_used);
    Itemset itemset;
    setUsingNodes((TrieNode*)root, itemset, n_used);

//    cout << "n_used: " << n_used << endl;

    switch (wipe_type) {
        case Subnodes:
            computeSubNodes((TrieNode *) root);
//             cout << "is subnodes hierarchy consistent : " << isConsistent((TrieNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            break;
        case Recall:
            // cout << "is reuse hierarchy consistent : " << isConsistent((TrieLtdNode*)root) << endl;
//            sort(heap.begin(), heap.end(), sortDecOrder);
            sort(heap.begin(), heap.end(), sortReuseDecOrder);
            break;
        default: // All. this block is used when the above if is commented.
            computeSubNodes((TrieNode *) root);
            // cout << "is subnodes hierarchy consistent for all wipe : " << isConsistent((TrieLtdNode*)root) << endl;
            sort(heap.begin(), heap.end(), sortDecOrder);
            n_del = heap.size();
    }

//     cout << "cachesize before wipe = " << getCacheSize() << endl;
    int counter = 0;
    for (auto it = heap.rbegin(); it != heap.rend(); it++) {

        // the node to remove
        TrieNode *node_del = it->first;
//        cout << "node_del:";
//        printItemset(it->second, true, false);
//        if(it->second.size() == 4 and it->second.at(0) == 16 and it->second.at(1) == 40 and it->second.at(2) == 43 and it->second.at(3) == 46) {
//            if (node_del->data) {
//                cout << " " << node_del << " " << node_del->data->test;// << endl;
//                if (node_del->data->left and node_del->data->right)cout << " child l:" << node_del->data->left << " child r:" << node_del->data->right << endl;
//            }
//            if (node_del->data and node_del->data->left) {
//                for (auto p: ((TrieNode*)node_del->data->left)->search_parents) {
//                    cout << p.first << ",";
//                }
//                cout << endl;
//            }
//        }



        // stop condition
        if (counter == n_del or node_del->is_used) break;

        // if the node to delete found its best solution, remove its pointer its left and right children
        if ( node_del->data != nullptr and node_del->data->left != nullptr ){
            TrieNode* node_del_left = ((TrieNode*)(node_del->data->left));
            TrieNode* node_del_right = ((TrieNode*)(node_del->data->right));
            node_del_left->search_parents.erase(std::find_if(node_del_left->search_parents.begin(), node_del_left->search_parents.end(), [node_del](const pair<TrieNode*,Itemset> &look) { return look.first == node_del; }));
            node_del_right->search_parents.erase(std::find_if(node_del_right->search_parents.begin(), node_del_right->search_parents.end(), [node_del](const pair<TrieNode*,Itemset> &look) { return look.first == node_del; }));
        }

//        if(it->second.size() == 4 and it->second.at(0) == 16 and it->second.at(1) == 40 and it->second.at(2) == 43 and it->second.at(3) == 46) {
//            if (node_del->data and node_del->data->left) {
//                for (auto p: ((TrieNode*)node_del->data->left)->search_parents) {
//                    cout << p.first << ",";
//                }
//                cout << endl;
//            }
//
//        }

        // remove from its search space parents, the fact that the node to delete is their best solution
        for (auto search_parent_node: node_del->search_parents) {

//            cout << "parent:";
//            printItemset(search_parent_node.second, true, false);
////            cout << " " << search_parent_node.first << "  ";
//            cout << " " << search_parent_node.first << endl;

//            if(search_parent_node.second.size() == 4 and search_parent_node.second.at(0) == 16 and search_parent_node.second.at(1) == 40 and search_parent_node.second.at(2) == 43 and search_parent_node.second.at(3) == 46) {
//                if (search_parent_node.first and search_parent_node.first->data) {
//                    cout << "par(" << search_parent_node.first->data->test << ")" << endl;
//                }
//                else cout << "par(null)" << endl;
//
//            }

            if (search_parent_node.first != nullptr and search_parent_node.first->data != nullptr and ( search_parent_node.first->data->left == node_del or search_parent_node.first->data->right == node_del ) ) {




                // remove the fact that the parent found the best solution
                if ( search_parent_node.first->data->error < FLT_MAX) {
                    search_parent_node.first->data->lowerBound = search_parent_node.first->data->error; // set the error as lb to help the re-computation
                    search_parent_node.first->data->error = FLT_MAX;
                    if (search_parent_node.first->data->test >= 0) search_parent_node.first->data->test = (search_parent_node.first->data->test + 1) * -1; // keep the best attribute in order to explore it first during the re-computation
                }

                // inform the corresponding left or right node (e.g. A will inform not A) that their parent won't recognize them anymore
                TrieNode* left_node = ((TrieNode*)search_parent_node.first->data->left);
                TrieNode* right_node = ((TrieNode*)search_parent_node.first->data->right);
                if ( search_parent_node.first->data->left == node_del )
                    right_node->search_parents.erase(std::find_if(right_node->search_parents.begin(), right_node->search_parents.end(), [search_parent_node](const pair<TrieNode*,Itemset> &look) { return look.first == search_parent_node.first; }));
                if ( search_parent_node.first->data->right == node_del )
                    left_node->search_parents.erase(std::find_if(left_node->search_parents.begin(), left_node->search_parents.end(), [search_parent_node](const pair<TrieNode*,Itemset> &look) { return look.first == search_parent_node.first; }));

                // invalidate the parent children
                search_parent_node.first->data->left = nullptr;
                search_parent_node.first->data->right = nullptr;

                // retro-propagate the information
                retroPropagate(search_parent_node.first);

            }
        }

        /*// remove from its trie parent the fact that the node to delete is its best solution
        auto trie_parent_node = node_del->trie_parent;
        if (trie_parent_node != nullptr and trie_parent_node->data != nullptr and
            ( trie_parent_node->data->left == node_del or
             trie_parent_node->data->right == node_del)) {

            trie_parent_node->data->test = -trie_parent_node->data->test;
            trie_parent_node->data->left = nullptr;
            trie_parent_node->data->right = nullptr;
            if (trie_parent_node->data->error < FLT_MAX) {
                trie_parent_node->data->lowerBound = trie_parent_node->data->error;
                trie_parent_node->data->error = FLT_MAX;
            }

        }*/

        // remove the edge bringing to the node
        node_del->trie_parent->edges.erase(find_if(node_del->trie_parent->edges.begin(), node_del->trie_parent->edges.end(), [node_del](const TrieEdge &look) { return look.subtrie == node_del; }));

        // remove the node
        delete node_del;
        heap.pop_back();
        counter++;
//        cout << endl;
    }
//    cout << endl;
//     cout << "cachesize after wipe = " << getCacheSize() << endl;
}

void Cache_Trie::setOptimalNodes(TrieNode *node, int& n_used) {
    if (node->data->left != nullptr) {
        ((TrieNode*)node->data->left)->is_used = true;
        ++n_used;
        setOptimalNodes((TrieNode*)node->data->left, n_used);

        ((TrieNode*)node->data->right)->is_used = true;
        ++n_used;
        setOptimalNodes((TrieNode*)node->data->right, n_used);
    }
}

void Cache_Trie::setUsingNodes(TrieNode *node, Itemset& itemset, int& n_used) {
    if (node and node->data and node->data->curr_test != -1) {
        Itemset itemset1 = addItem(itemset, item(node->data->curr_test, NEG_ITEM));
        auto node1 = (TrieNode *)getandSet(itemset1, n_used);
        setUsingNodes(node1, itemset1, n_used);

        Itemset itemset2 = addItem(itemset, item(node->data->curr_test, POS_ITEM));
        auto node2 = (TrieNode *)getandSet(itemset2, n_used);
        setUsingNodes(node2, itemset2, n_used);
    }
}

int Cache_Trie::computeSubNodes(TrieNode *node) {
    node->n_subnodes = 0;
    if (node->edges.empty()) { return 0; }
    for (auto &edge: node->edges) node->n_subnodes += 1 + computeSubNodes(edge.subtrie);
    return node->n_subnodes;
}

bool Cache_Trie::isLoadConsistent(TrieNode *node, Itemset itemset) {
    TrieNode *parent = node;
    if (parent->edges.empty()) return true;
    auto loadSum = [parent]() {
        int sum = 1;
        for (const auto &edge: parent->edges) sum += edge.subtrie->count_opti_path;
        return sum;
    };
    if (parent != root and not(parent->count_opti_path == 0 and loadSum() == 1) and
        parent->count_opti_path < loadSum()) {
        cout << "par_ite:";
        printItemset(itemset, true, false);
        cout << " par:" << parent->count_opti_path << " load_sum:" << loadSum() << endl;
        return false;
    }
    return all_of(parent->edges.begin(), parent->edges.end(), [itemset, this](const TrieEdge &edge) {
        TrieNode *child = edge.subtrie;
        Itemset itemset1 = itemset;
        itemset1.push_back(edge.item);
        if (not isLoadConsistent(child, itemset1)) return false;
        return true;
    });
}

bool Cache_Trie::isNonNegConsistent(TrieNode *node) {
    return all_of(node->edges.begin(), node->edges.end(), [this](const TrieEdge &edge) {
        TrieNode *child = edge.subtrie;
        if (child->count_opti_path < 0 or not isNonNegConsistent(child)) return false;
        return true;
    });
}


bool Cache_Trie::isConsistent(TrieNode *node) {
    return all_of(node->edges.begin(), node->edges.end(), [node, this](const TrieEdge &edge) {
        TrieNode *parent = node, *child = edge.subtrie;
        if (
                ((wipe_type == Subnodes || wipe_type == All) && child->n_subnodes >= parent->n_subnodes) or
                (wipe_type == Recall && (child->n_reuse > parent->n_reuse ||
                                         (child->n_reuse == parent->n_reuse && child->depth <= parent->depth))) or
                (not isConsistent(child))
                )
            return false;
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
    if (inc) Logger::showMessage("increasing load of itemset: ");
    else Logger::showMessage("decreasing load of itemset: ");
    printItemset(itemset);
    int i = 0;
    for (const auto item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end() && geqEdge_it->item == item) { // item found
            cur_node = geqEdge_it->subtrie;
            if (inc) cur_node->count_opti_path++;
            else cur_node->count_opti_path--;

            for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
            Logger::showMessage(":(", cur_node->count_opti_path, ") -- ");

            if (cur_node->count_opti_path < 0) {
                cout << "load itemset: ";
                for (int j = 0; j <= i; ++j) cout << itemset.at(j) << ",";
                cout << " " << cur_node->count_opti_path << endl;
            }
        }
        i++;
    }
    Logger::showMessageAndReturn("");
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
    for (const auto &item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it != cur_node->edges.end() && geqEdge_it->item == item) { // item found
            cur_node = geqEdge_it->subtrie;
            for (int j = 0; j <= i; ++j) Logger::showMessage(itemset.at(j), ",");
            Logger::showMessage(":(", cur_node->count_opti_path, ") -- ");
        }
        i++;
    }
    cout << endl;
}


void Cache_Trie::printSubTreeLoad(Itemset &itemset, Item firstItem, Item secondItem, bool inc) {
    for (auto item: {firstItem, secondItem}) {

        if (item == -1) {
            if (item == secondItem) Itemset().swap(itemset);
            continue;
        }

        Itemset child_itemset = addItem(itemset, item);
        if (item == secondItem)
            Itemset().swap(itemset); // when we build both branches itemsets, we don't need the parent anymore
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
    for (const auto &item: itemset) {
        cur_node = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge)->subtrie;
        cur_node->count_opti_path += value;
    }
}

//void Cache_Trie::updateParents(Node *best_, Node *left, Node *right) {
void Cache_Trie::updateParents(Node *best_, Node *left, Node *right, Itemset itemset) {
    TrieNode *best = (TrieNode*) best_, *new_left = (TrieNode*) left, *new_right = (TrieNode*) right;
    TrieNode *old_left = ((TrieNode*)best->data->left), *old_right = ((TrieNode*)best->data->right);

    if ( best->data->test >= 0 ) {
        if (old_left != nullptr) {
            auto it = std::find_if(old_left->search_parents.begin(), old_left->search_parents.end(), [best](const pair<TrieNode*,Itemset> &look) { return look.first == best; });
//            auto it = std::find(old_left->search_parents.begin(), old_left->search_parents.end(), best);
            if (it != old_left->search_parents.end())
                old_left->search_parents.erase(it);
        }

        if (old_right != nullptr) {
            auto it = std::find_if(old_right->search_parents.begin(), old_right->search_parents.end(), [best](const pair<TrieNode*,Itemset> &look) { return look.first == best; });
//            auto it = std::find(old_right->search_parents.begin(), old_right->search_parents.end(), best);
            if (it != old_right->search_parents.end())
                old_right->search_parents.erase(it);
        }

    }

//    new_left->search_parents.push_back(best);
//    new_right->search_parents.push_back(best);
    new_left->search_parents.insert(make_pair(best, itemset));
    new_right->search_parents.insert(make_pair(best, itemset));
}

void Cache_Trie::retroPropagate(TrieNode *node) {

    for (auto parent_el: node->search_parents) {
        auto parent = parent_el.first;
        if (parent != nullptr and parent->data != nullptr) {
            if (parent->data->test >= 0) parent->data->test = (parent->data->test + 1) * -1;
        }
        if (parent != nullptr) {
            retroPropagate(parent);
        }
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
