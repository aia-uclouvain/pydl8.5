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
/*bool sortDecOrder( TrieNode *&pair1,  TrieNode *&pair2) {
    const auto node1 = pair1, node2 = pair2;
    if (node1->is_used and not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used and node2->is_used) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}*/

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

// in case of heap decreasing comparator for sort provides a min-heap
//bool minHeapOrder(const pair<TrieLtdNode*,vector<Item>> &pair1, const pair<TrieLtdNode*,vector<Item>> &pair2) {
/*bool minHeapOrder( TrieNode *&pair1,  TrieNode *&pair2) {
    return sortDecOrder(pair1, pair2);
}*/

bool sortDecOrder(pair<TrieNode *, Itemset> &pair1, pair<TrieNode *, Itemset> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->is_used and not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used and node2->is_used) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}

bool sortReuseDecOrder(pair<TrieNode *, Itemset> &pair1, pair<TrieNode *, Itemset> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->is_used && not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used && node2->is_used) return false; // same for the node2
    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth)
        return node1->depth < node2->depth; // depth from bigger to lower, so sorted in creasing order
//    if (node1->n_reuse == node2->n_reuse && node1->depth != node2->depth) return node1->support > node2->support;
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}

bool minHeapOrder(pair<TrieNode *, Itemset> &pair1, pair<TrieNode *, Itemset> &pair2) {
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

Node *Cache_Trie::getandSet(const Itemset &itemset, int &n_used) {
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
            Itemset its;
            for (int j = 0; j <= i; ++j) {
                int e = itemset.at(j);
                its.push_back(e);
            }
            heap.push_back(make_pair(child_node, its));
            //heap.push_back(child_node);

//            if(its.size() == 6 and its.at(0) == 0 and its.at(1) == 2 and its.at(2) == 11 and its.at(3) == 12 and its.at(4) == 17 and its.at(5) == 28) {
//                cout << "created " << child_node << endl;
//                if(itemset.size() == 6 and itemset.at(0) == 0 and itemset.at(1) == 2 and itemset.at(2) == 11 and itemset.at(3) == 12 and itemset.at(4) == 17 and itemset.at(5) == 28) {
//                    cout << "meme" << endl;
//                }
//                else cout << "chemin" << endl;
//            }
//
//            if(its.size() == 5 and its.at(0) == 0 and its.at(1) == 2 and its.at(2) == 11 and its.at(3) == 12 and its.at(4) == 28) {
//                cout << "- parent created " << child_node << endl;
//                if(itemset.size() == 5 and itemset.at(0) == 0 and itemset.at(1) == 2 and itemset.at(2) == 11 and itemset.at(3) == 12 and itemset.at(4) == 28) {
//                    cout << "meme" << endl;
//                }
//                else cout << "chemin" << endl;
//            }

            child_node->trie_parent = parent_node;
        }

        TrieEdge newedge{itemset[i], child_node};
        if (i == pos) parent_node->edges.insert(geqEdge_it, newedge);
        else parent_node->edges.push_back(newedge); // new node added so add the edge without checking its place
        cachesize++;
        child_node->depth = i + 1;
        parent_node = child_node;
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

//    Logger::showMessage("increasing load of itemset: ");
//    printItemset(itemset);

    vector<TrieEdge>::iterator geqEdge_it;
    for (int i = 0; i < itemset.size(); ++i) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() or geqEdge_it->item != itemset[i]) { // the item does not exist

            if (getCacheSize() + itemset.size() - i > maxcachesize && maxcachesize > NO_CACHE_LIMIT) {
                Logger::showMessageAndReturn("wipe launched");
//                cout << "wipe" << endl;
                wipe();
                geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
//                cout << "done" << endl;
                Logger::showMessageAndReturn("wipe done");
            }

            // create path representing the part of the itemset not yet present in the trie.
            TrieNode *last_inserted_node = addNonExistingItemsetPart(itemset, i, geqEdge_it, cur_node);
            Logger::showMessageAndReturn("");

            return {last_inserted_node, true};
        } else {
            if (i == 0) cur_node->n_reuse++; // root node
            cur_node = geqEdge_it->subtrie;
            cur_node->n_reuse++;
            cur_node->depth = i + 1;
        }
    }
    Logger::showMessageAndReturn("");

    if (cur_node->data == nullptr) return {cur_node, true};
    else return {cur_node, false};
}

void Cache_Trie::setOptimalNodes(TrieNode *node, int &n_used) {
    if (node->data->left != nullptr) {
        ((TrieNode *) node->data->left)->is_used = true;
        ++n_used;
        setOptimalNodes((TrieNode *) node->data->left, n_used);

        ((TrieNode *) node->data->right)->is_used = true;
        ++n_used;
        setOptimalNodes((TrieNode *) node->data->right, n_used);
    }
}

void Cache_Trie::setUsingNodes(TrieNode *node, Itemset &itemset, int &n_used) {
    if (node and node->data and node->data->curr_test != -1) {
        Itemset itemset1 = addItem(itemset, item(node->data->curr_test, NEG_ITEM));
        auto node1 = (TrieNode *) getandSet(itemset1, n_used);
        setUsingNodes(node1, itemset1, n_used);

        Itemset itemset2 = addItem(itemset, item(node->data->curr_test, POS_ITEM));
        auto node2 = (TrieNode *) getandSet(itemset2, n_used);
        setUsingNodes(node2, itemset2, n_used);
    }
}

int Cache_Trie::computeSubNodes(TrieNode *node) {
    node->n_subnodes = 0;
    if (node->edges.empty()) { return 0; }
    for (auto &edge: node->edges) node->n_subnodes += 1 + computeSubNodes(edge.subtrie);
    return node->n_subnodes;
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


//void Cache_Trie::updateParents(Node *best_, Node *left, Node *right) {
void Cache_Trie::updateParents(Node *best_, Node *left, Node *right, Itemset itemset) {
    TrieNode *best = (TrieNode *) best_, *new_left = (TrieNode *) left, *new_right = (TrieNode *) right;
    TrieNode *old_left = ((TrieNode *) best->data->left), *old_right = ((TrieNode *) best->data->right);

    if (best->data->test >= 0) {
        if (old_left != nullptr) {
            auto it = std::find_if(old_left->search_parents.begin(), old_left->search_parents.end(),
                                   [best](const pair<TrieNode *, Itemset> &look) { return look.first == best; });
//            auto it = std::find(old_left->search_parents.begin(), old_left->search_parents.end(), best);
            if (it != old_left->search_parents.end())
                old_left->search_parents.erase(it);
        }

        if (old_right != nullptr) {
            auto it = std::find_if(old_right->search_parents.begin(), old_right->search_parents.end(),
                                   [best](const pair<TrieNode *, Itemset> &look) { return look.first == best; });
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
        if (parent == root) continue;
        if (parent != nullptr and parent->data != nullptr) {
            if (parent->data->test >= 0) parent->data->test = (parent->data->test + 1) * -1;
            parent->data->lowerBound = max(parent->data->lowerBound, parent->data->error);
//            parent->data->error = FLT_MAX;
        }
        if (parent != nullptr) {
            retroPropagate(parent);
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

    int n_del = (int) (maxcachesize * wipe_factor);
    int n_used = 0;
    setOptimalNodes((TrieNode *) root, n_used);
    Itemset itemset;
    setUsingNodes((TrieNode *) root, itemset, n_used);

//    cout << "n_used: " << n_used << endl;

    // sort the nodes in the cache based on the heuristic used to remove them
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
//        printItemset(it->second, true, true);
//        cout << "yopooo" << endl;
//        if(it->second.size() == 6 and it->second.at(0) == 0 and it->second.at(1) == 2 and it->second.at(2) == 11 and it->second.at(3) == 12 and it->second.at(4) == 17 and it->second.at(5) == 28) {
//            cout << "node_del:";
//            printItemset(it->second, true, false);
//            if (node_del->data) {
//                cout << " " << node_del << " " << node_del->data->test << " ";// << endl;
////                if (node_del->data->left and node_del->data->right)
//                    cout << " child l:" << node_del->data->left << " child r:" << node_del->data->right << endl;
//            }
//            cout << " parents: ";
//            for (auto p: node_del->search_parents) {
//                printItemset(p.second, true, false);
//                cout << " (" << p.first->data->error << "), ";
//            }
//            cout << endl;
////            if (node_del->data and node_del->data->left) {
////                for (auto p: ((TrieNode*)node_del->data->left)->search_parents) {
////                    cout << p.first << ",";
////                }
////                cout << endl;
////            }
//        }

//        if(it->second.size() == 5 and it->second.at(0) == 0 and it->second.at(1) == 2 and it->second.at(2) == 11 and it->second.at(3) == 12 and it->second.at(4) == 28) {
//            cout << "- parent node_del:";
//            printItemset(it->second, true, true);
////            cout << "best child: " << it->first->data->test << endl;
//        }

        // stop condition
        if (counter == n_del or node_del->is_used) break;

        // remove from its children, the fact that the node to delete is one of their parents
        if (node_del->data != nullptr and node_del->data->left != nullptr) {
            TrieNode *left_child = ((TrieNode *) (node_del->data->left));
            TrieNode *right_child = ((TrieNode *) (node_del->data->right));
            left_child->search_parents.erase(std::find_if(left_child->search_parents.begin(), left_child->search_parents.end(), [node_del](const pair<TrieNode *, Itemset> &look) { return look.first == node_del; }));
            right_child->search_parents.erase(std::find_if(right_child->search_parents.begin(), right_child->search_parents.end(), [node_del](const pair<TrieNode *, Itemset> &look) { return look.first == node_del; }));
        }

        // remove from its parents, the fact that the node to delete is their best solution
        for (auto parent_node: node_del->search_parents) {
//            cout << "parrra:";
//            printItemset(parent_node.second, true, true);
//            if(parent_node.second.size() == 3 and
//            (
//                    (parent_node.second.at(0) == 11 and parent_node.second.at(1) == 14 and parent_node.second.at(2) == 28) or
//                    (parent_node.second.at(0) == 12 and parent_node.second.at(1) == 14 and parent_node.second.at(2) == 32) or
//                    (parent_node.second.at(0) == 2 and parent_node.second.at(1) == 6 and parent_node.second.at(2) == 17)
//                    )
//            ) {
//                cout << "parent:";
//                printItemset(parent_node.second, true, true);
//                printItemset(it->second, true, true);
//                cout << parent_node.first << endl;
//                cout << parent_node.first->data << endl;
//                cout << parent_node.first->data->test << endl;
//                cout << parent_node.first->data->test << " " << parent_node.first->data->left << " " << parent_node.first->data->right << endl;
//            }

            if (parent_node.first != nullptr and parent_node.first->data != nullptr and (parent_node.first->data->left == node_del or parent_node.first->data->right == node_del) ) {

                // remove the fact that the parent found the best solution
                if (parent_node.first->data->error < FLT_MAX) {
                    parent_node.first->data->lowerBound = parent_node.first->data->error; // set the error as lb to help the re-computation
                    parent_node.first->data->error = FLT_MAX;
//                    if(parent_node.second.size() == 5 and parent_node.second.at(0) == 0 and parent_node.second.at(1) == 2 and parent_node.second.at(2) == 11 and parent_node.second.at(3) == 12 and parent_node.second.at(4) == 28) {
//                        cout << "- for parent, child to del is:";
//                        printItemset(it->second, true, true);
//                        cout << "best child of parent: " << parent_node.first->data->test << endl;
//                    }
                    if (parent_node.first->data->test >= 0)
                        parent_node.first->data->test = (parent_node.first->data->test + 1) * -1; // keep the best attribute in order to explore it first during the re-computation
                }

                // inform the corresponding left or right node (e.g. A will inform not A) that their parent won't recognize them anymore
                TrieNode *left_node = ((TrieNode *) parent_node.first->data->left);
                TrieNode *right_node = ((TrieNode *) parent_node.first->data->right);
                if (parent_node.first->data->left == node_del)
                    right_node->search_parents.erase(
                            std::find_if(right_node->search_parents.begin(), right_node->search_parents.end(),
                                         [parent_node](const pair<TrieNode *, Itemset> &look) {
                                             return look.first == parent_node.first;
                                         }));
                if (parent_node.first->data->right == node_del)
                    left_node->search_parents.erase(
                            std::find_if(left_node->search_parents.begin(), left_node->search_parents.end(),
                                         [parent_node](const pair<TrieNode *, Itemset> &look) {
                                             return look.first == parent_node.first;
                                         }));

                // invalidate the parent children
                parent_node.first->data->left = nullptr;
                parent_node.first->data->right = nullptr;

                // retro-propagate the information
//                retroPropagate(parent_node.first);

            }
        }
//        cout << "fin" << endl;

        // remove the edge bringing to the node
        node_del->trie_parent->edges.erase(
                find_if(node_del->trie_parent->edges.begin(), node_del->trie_parent->edges.end(),
                        [node_del](const TrieEdge &look) { return look.subtrie == node_del; }));

        // remove the node
        delete node_del;
        heap.pop_back();
        counter++;
    }
//     cout << "cachesize after wipe = " << getCacheSize() << endl;
}
