#include "cache_hash_cover.h"
#include "nodeDataManager_Cover.h"

using namespace std;

Cache_Hash_Cover::Cache_Hash_Cover(Depth maxdepth, WipeType wipe_type, int maxcachesize, float wipe_factor) :
        Cache(maxdepth, wipe_type, maxcachesize), wipe_factor( wipe_factor) {
    root = new HashCoverNode();
    store = new unordered_map<MyCover, HashCoverNode *>[maxdepth];
    if (write_stats) myfile.open (dataname + "_d" + std::to_string(maxdepth) + "_h_cover.txt", ios::out);

//    if (this->maxcachesize > NO_CACHE_LIMIT) {
////        heap = new vector<pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Itemset>>[maxdepth];
//        heap = new vector<const unordered_map<MyCover, HashCoverNode*>::iterator*>[maxdepth];
////    heap = new vector<HashCoverNode*>[maxdepth];
//        for (int i = 0; i < maxdepth; ++i) {
//            heap[i].reserve(maxcachesize);
//        }
//    }

}

pair<Node *, bool> Cache_Hash_Cover::insert(NodeDataManager *nodeDataManager, int depth, Itemset itemset) {
//    for (int i = 0; i < maxdepth; ++i) {
//        for (auto &e : heap[i]) {
//            const unordered_map<MyCover, HashCoverNode*>::iterator* a = e.first;
//            unordered_map<MyCover, HashCoverNode*>::iterator b = *a;
//            if (b->second == nullptr) cout << "papapaap" << endl;
//        }
//    }
    if (depth == 0) {
        cachesize++;
        return {root, true};
    } else {
//        cout << "a" << endl;
        if (maxcachesize > NO_CACHE_LIMIT and getCacheSize() >= maxcachesize) {
//            cout << "wipe" << endl;
            wipe();
        }
//        cout << "a" << endl;
        auto *node = new HashCoverNode();
//        cout << "a" << endl;
        // pair<const unordered_map<MyCover, HashCoverNode*>::iterator&, bool> info = store[depth-1].insert({MyCover(nodeDataManager->cover), node});
        auto info = store[depth-1].insert({MyCover(nodeDataManager->cover), node});
//        pair<unordered_map<MyCover, HashCoverNode *>::iterator, bool> info = store[depth-1].insert({MyCover(nodeDataManager->cover), node});
//        cout << "a" << endl;
        if (not info.second) { // if node already exists
//            cout << "here" << endl;
            delete node;
//            info.first->second->n_reuse++;
        }
        else {
//            cout << "here1" << endl;
//            for (auto e : heap[depth - 1]) {
//                if ((*(e.first))->second == nullptr) cout << "papapaap" << endl;
//            }
//            if (maxcachesize > NO_CACHE_LIMIT) {
//                // THERE IS A PROBLEM NOT YET IDENTIFIED WITH THE NEXT LINE
//                heap[depth-1].push_back(&(info.first));
//            }
//            if (heap[depth-1].back()->operator->()->second == nullptr) cout << "nulo" << endl;

//            if (maxcachesize > NO_CACHE_LIMIT) heap[depth-1].push_back(make_pair(&(info.first), itemset));
//            if (heap[depth-1].back().first->operator->()->second == nullptr) cout << "nulo" << endl;

//            cout << "coulo " << heap[depth-1].back().first << " ";
//            printItemset(heap[depth-1].back().second, true, true);
//            if (maxcachesize > NO_CACHE_LIMIT) heap[depth-1].push_back(make_pair(info.first->second, itemset));
//            if (maxcachesize > NO_CACHE_LIMIT) heap[depth-1].push_back(info.first->second);
        }

        if (write_stats) {
            std::chrono::time_point<std::chrono::high_resolution_clock> c_time = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<float>(c_time - last_time).count() >= write_gap) {
                myfile << std::chrono::duration<float>(c_time - startTime).count() << "," << getCacheSize() << "\n";
                myfile.flush();
                last_time = c_time;
            }
        }

        return {info.first->second, info.second};
        //if (cachesize >= maxcachesize && maxcachesize > 0) wipe(root);
    }
}

Node *Cache_Hash_Cover::get(NodeDataManager *nodeDataManager, int depth) {
    auto it = store[depth-1].find(MyCover(nodeDataManager->cover));
    if (it != store[depth-1].end()) return it->second;
    else return nullptr;
}

int Cache_Hash_Cover::getCacheSize() {
    int val = 0;
    for (int i = 0; i < maxdepth; ++i) {
        val += store[i].size();
    }
    return val;
}

/*bool sortReuseDecOrder(HashCoverNode *&node1, HashCoverNode *&node2) {
    if (node1->is_used && not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used && node2->is_used) return false; // same for the node2
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}*/

/*bool sortReuseDecOrder(pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Itemset> &pair1, pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Itemset> &pair2) {
    const auto node1 = (*(pair1.first))->second, node2 = (*(pair2.first))->second;
    if (node1->is_used && not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used && node2->is_used) return false; // same for the node2
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}*/

/*bool sortReuseDecOrder(const unordered_map<MyCover, HashCoverNode*>::iterator *&pair1, const unordered_map<MyCover, HashCoverNode*>::iterator *&pair2) {
    const auto node1 = (*pair1)->second, node2 = (*pair2)->second;
    if (node1->is_used && not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used && node2->is_used) return false; // same for the node2
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}*/

/*bool sortReuseDecOrder(pair<HashCoverNode *, Itemset> &pair1, pair<HashCoverNode *, Itemset> &pair2) {
    const auto node1 = pair1.first, node2 = pair2.first;
    if (node1->is_used && not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used && node2->is_used) return false; // same for the node2
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}*/

/*void Cache_Hash_Cover::setOptimalNodes(HashCoverNode *node, int &n_used) {
    if (((CoverNodeData*)node->data)->left != nullptr) {
        ((CoverNodeData*)node->data)->left->is_used = true;
        ++n_used;
        setOptimalNodes((HashCoverNode *) ((CoverNodeData*)node->data)->left, n_used);

        ((HashCoverNode *) ((CoverNodeData*)node->data)->right)->is_used = true;
        ++n_used;
        setOptimalNodes((HashCoverNode *) ((CoverNodeData*)node->data)->right, n_used);
    }
}*/

/*void Cache_Hash_Cover::setUsingNodes(HashCoverNode *node, int &n_used) {
    if (node and node->data and ((CoverNodeData*)node->data)->curr_left != nullptr) {
        ((CoverNodeData*)node->data)->curr_left->is_used = true;
        setUsingNodes((HashCoverNode *) ((CoverNodeData*)node->data)->curr_left, n_used);
    }

    if (node and node->data and ((CoverNodeData*)node->data)->curr_right != nullptr) {
        ((CoverNodeData*)node->data)->curr_right->is_used = true;
        setUsingNodes((HashCoverNode *) ((CoverNodeData*)node->data)->curr_right, n_used);
    }
}*/

/*void Cache_Hash_Cover::updateParents(Node *best_, Node *left, Node *right, Itemset itemset) {
    auto *best = (HashCoverNode *) best_, *new_left = (HashCoverNode *) left, *new_right = (HashCoverNode *) right;
    auto *old_left = ((HashCoverNode *) best->data->left), *old_right = ((HashCoverNode *) best->data->right);
    Attribute  old_test = best->data->test;

//    if (old_test >= 0 and old_test != INT32_MAX) {
//        if (old_left != nullptr) {
//            old_left->search_parents.erase(best);
//        }
//
//        if (old_right != nullptr) {
//            old_right->search_parents.erase(best);
//        }
//
//    }
    if (best->data->test >= 0) {
        if (old_left != nullptr) {
            auto it = std::find_if(old_left->search_parents.begin(), old_left->search_parents.end(),
                                   [best](const pair<HashCoverNode *, Itemset> &look) { return look.first == best; });
//            auto it = std::find(old_left->search_parents.begin(), old_left->search_parents.end(), best);
            if (it != old_left->search_parents.end())
                old_left->search_parents.erase(it);
        }

        if (old_right != nullptr) {
            auto it = std::find_if(old_right->search_parents.begin(), old_right->search_parents.end(),
                                   [best](const pair<HashCoverNode *, Itemset> &look) { return look.first == best; });
//            auto it = std::find(old_right->search_parents.begin(), old_right->search_parents.end(), best);
            if (it != old_right->search_parents.end())
                old_right->search_parents.erase(it);
        }

    }

//    new_left->search_parents.insert(best);
//    new_right->search_parents.insert(best);
    new_left->search_parents.insert(make_pair(best, itemset));
    new_right->search_parents.insert(make_pair(best, itemset));
}*/

//void Cache_Hash_Cover::wipe() {
//
//    int n_del = (int) (maxcachesize * wipe_factor);
//    int n_used = 0;
//    setOptimalNodes((HashCoverNode *) root, n_used);
//    setUsingNodes((HashCoverNode *) root, n_used);
//
//    cout << "n_used: " << n_used << endl;
//
//    // sort the nodes in the cache based on the heuristic used to remove them
//    switch (wipe_type) {
//        case Subnodes:
////            computeSubNodes((HashCoverNode *) root);
////             cout << "is subnodes hierarchy consistent : " << isConsistent((TrieNode*)root) << endl;
//            for (int i = 0; i < maxdepth; ++i) {
//                sort(heap[i].begin(), heap[i].end(), sortReuseDecOrder);
//            }
//
//            break;
//        case Recall:
//            // cout << "is reuse hierarchy consistent : " << isConsistent((TrieLtdNode*)root) << endl;
////            sort(heap.begin(), heap.end(), sortDecOrder);
//            for (int i = 0; i < maxdepth; ++i) {
//                sort(heap[i].begin(), heap[i].end(), sortReuseDecOrder);
//            }
//            break;
//        default: // All. this block is used when the above if is commented.
////            computeSubNodes((HashCoverNode *) root);
//            // cout << "is subnodes hierarchy consistent for all wipe : " << isConsistent((TrieLtdNode*)root) << endl;
//            for (int i = 0; i < maxdepth; ++i) {
//                sort(heap[i].begin(), heap[i].end(), sortReuseDecOrder);
//            }
////            n_del = heap.size();
//    }
//
//     cout << "cachesize before wipe = " << getCacheSize() << endl;
//    cout << n_del << " " << maxcachesize << " " << wipe_factor << endl;
//    int counter = 0;
//    bool n_del_reached = false;
//    for (int depth = maxdepth - 1; depth >= 0; --depth) {
//        if (n_del_reached) {
//            cout << "fi" << endl;
//            break;
//        }
//        for (auto it = heap[depth].rbegin(); it != heap[depth].rend(); it++) {
//
//            // the node to remove
//            HashCoverNode *node_del = (*(*it))->second;
////            HashCoverNode *node_del = (*(it->first))->second;
//
////        cout << "node_del:";
////        printItemset(it->second, true, true);
////        cout << "yopooo" << endl;
////        if(it->second.size() == 6 and it->second.at(0) == 0 and it->second.at(1) == 2 and it->second.at(2) == 11 and it->second.at(3) == 12 and it->second.at(4) == 17 and it->second.at(5) == 28) {
////            cout << "node_del:";
////            printItemset(it->second, true, false);
////            if (node_del->data) {
////                cout << " " << node_del << " " << node_del->data->test << " ";// << endl;
//////                if (node_del->data->left and node_del->data->right)
////                    cout << " child l:" << node_del->data->left << " child r:" << node_del->data->right << endl;
////            }
////            cout << " parents: ";
////            for (auto p: node_del->search_parents) {
////                printItemset(p.second, true, false);
////                cout << " (" << p.first->data->error << "), ";
////            }
////            cout << endl;
//////            if (node_del->data and node_del->data->left) {
//////                for (auto p: ((TrieNode*)node_del->data->left)->search_parents) {
//////                    cout << p.first << ",";
//////                }
//////                cout << endl;
//////            }
////        }
//
////        if(it->second.size() == 5 and it->second.at(0) == 0 and it->second.at(1) == 2 and it->second.at(2) == 11 and it->second.at(3) == 12 and it->second.at(4) == 28) {
////            cout << "- parent node_del:";
////            printItemset(it->second, true, true);
//////            cout << "best child: " << it->first->data->test << endl;
////        }
//
//            // stop condition
//            if (counter == n_del) {
//                n_del_reached = true;
//                cout << "fo" << endl;
//                break;
//            }
//            if (node_del->is_used) {
//                if (depth == 0) {
//                    cout << "fu" << endl;
//                    break;
//                }
//                else continue;
//            }
//
//            // remove from its children, the fact that the node to delete is one of their parents
//            if (node_del->data != nullptr and ((CoverNodeData*)node_del->data)->left != nullptr) {
//                HashCoverNode *left_child = ((HashCoverNode *) (((CoverNodeData*)node_del->data)->left));
//                HashCoverNode *right_child = ((HashCoverNode *) (((CoverNodeData*)node_del->data)->right));
//                left_child->search_parents.erase(
//                        std::find_if(left_child->search_parents.begin(), left_child->search_parents.end(),
//                                     [node_del](const pair<HashCoverNode *, Itemset> &look) { return look.first == node_del; }));
//                right_child->search_parents.erase(
//                        std::find_if(right_child->search_parents.begin(), right_child->search_parents.end(),
//                                     [node_del](const pair<HashCoverNode *, Itemset> &look) { return look.first == node_del; }));
////                left_child->search_parents.erase(left_child->search_parents.find(node_del));
////                right_child->search_parents.erase(right_child->search_parents.find(node_del));
//            }
//
//            // remove from its parents, the fact that the node to delete is their best solution
//            for (auto parent_node: node_del->search_parents) {
////            cout << "parrra:";
////            printItemset(parent_node.second, true, true);
////            if(parent_node.second.size() == 3 and
////            (
////                    (parent_node.second.at(0) == 11 and parent_node.second.at(1) == 14 and parent_node.second.at(2) == 28) or
////                    (parent_node.second.at(0) == 12 and parent_node.second.at(1) == 14 and parent_node.second.at(2) == 32) or
////                    (parent_node.second.at(0) == 2 and parent_node.second.at(1) == 6 and parent_node.second.at(2) == 17)
////                    )
////            ) {
////                cout << "parent:";
////                printItemset(parent_node.second, true, true);
////                printItemset(it->second, true, true);
////                cout << parent_node.first << endl;
////                cout << parent_node.first->data << endl;
////                cout << parent_node.first->data->test << endl;
////                cout << parent_node.first->data->test << " " << parent_node.first->data->left << " " << parent_node.first->data->right << endl;
////            }
//
//                if (parent_node.first != nullptr and parent_node.first->data != nullptr and
//                    (((CoverNodeData*)parent_node.first->data)->left == node_del or ((CoverNodeData*)parent_node.first->data)->right == node_del)) {
//
//                    // remove the fact that the parent found the best solution
//                    if (parent_node.first->data->error < FLT_MAX) {
//                        parent_node.first->data->lowerBound = parent_node.first->data->error; // set the error as lb to help the re-computation
//                        parent_node.first->data->error = FLT_MAX;
////                    if(parent_node.second.size() == 5 and parent_node.second.at(0) == 0 and parent_node.second.at(1) == 2 and parent_node.second.at(2) == 11 and parent_node.second.at(3) == 12 and parent_node.second.at(4) == 28) {
////                        cout << "- for parent, child to del is:";
////                        printItemset(it->second, true, true);
////                        cout << "best child of parent: " << parent_node.first->data->test << endl;
////                    }
//                        if (parent_node.first->data->test >= 0)
//                            parent_node.first->data->test = (parent_node.first->data->test + 1) *
//                                                            -1; // keep the best attribute in order to explore it first during the re-computation
//                    }
//
//                    // inform the corresponding left or right node (e.g. A will inform not A) that their parent won't recognize them anymore
//                    HashCoverNode *left_node = ((HashCoverNode *) ((CoverNodeData*)parent_node.first->data)->left);
//                    HashCoverNode *right_node = ((HashCoverNode *) ((CoverNodeData*)parent_node.first->data)->right);
//                    if (((CoverNodeData*)parent_node.first->data)->left == node_del)
//                        right_node->search_parents.erase(
//                                std::find_if(right_node->search_parents.begin(), right_node->search_parents.end(),
//                                             [parent_node](const pair<HashCoverNode *, Itemset> &look) {
//                                                 return look.first == parent_node.first;
//                                             }));
//                    if (((CoverNodeData*)parent_node.first->data)->right == node_del)
//                        left_node->search_parents.erase(
//                                std::find_if(left_node->search_parents.begin(), left_node->search_parents.end(),
//                                             [parent_node](const pair<HashCoverNode *, Itemset> &look) {
//                                                 return look.first == parent_node.first;
//                                             }));
//
//                    // invalidate the parent children
//                    ((CoverNodeData*)parent_node.first->data)->left = nullptr;
//                    ((CoverNodeData*)parent_node.first->data)->right = nullptr;
//
//                    // retro-propagate the information
////                retroPropagate(parent_node.first);
//
//                }
//
//                /*if (parent_node != nullptr and parent_node->data != nullptr and
//                    (parent_node->data->left == node_del or parent_node->data->right == node_del)) {
//
//                    // remove the fact that the parent found the best solution
//                    if (parent_node->data->error < FLT_MAX) {
//                        parent_node->data->lowerBound = parent_node->data->error; // set the error as lb to help the re-computation
//                        parent_node->data->error = FLT_MAX;
////                    if(parent_node.second.size() == 5 and parent_node.second.at(0) == 0 and parent_node.second.at(1) == 2 and parent_node.second.at(2) == 11 and parent_node.second.at(3) == 12 and parent_node.second.at(4) == 28) {
////                        cout << "- for parent, child to del is:";
////                        printItemset(it->second, true, true);
////                        cout << "best child of parent: " << parent_node.first->data->test << endl;
////                    }
//                        if (parent_node->data->test >= 0)
//                            parent_node->data->test = (parent_node->data->test + 1) * -1; // keep the best attribute in order to explore it first during the re-computation
//                    }
//
//                    // inform the corresponding left or right node (e.g. A will inform not A) that their parent won't recognize them anymore
//                    HashCoverNode *left_node = ((HashCoverNode *) parent_node->data->left);
//                    HashCoverNode *right_node = ((HashCoverNode *) parent_node->data->right);
//                    if (parent_node->data->left == node_del)
//                        right_node->search_parents.erase(right_node->search_parents.find(parent_node));
//                    if (parent_node->data->right == node_del)
//                        left_node->search_parents.erase(left_node->search_parents.find(parent_node));
//
//                    // invalidate the parent children
//                    parent_node->data->left = nullptr;
//                    parent_node->data->right = nullptr;
//
//                    // retro-propagate the information
////                retroPropagate(parent_node.first);
//
//                }*/
//            }
////        cout << "fin" << endl;
//
//            // remove the node
//            delete node_del;
//            store[depth].erase((*(*it))->first);
////            store[depth].erase((*(it->first))->first);
//            heap[depth].pop_back();
//            counter++;
//        }
//    }
//     cout << "cachesize after wipe = " << getCacheSize() << endl;
//}

/*void Cache_Hash::updateSubTreeLoad(Array<Item> itemset, Item firstItem, Item secondItem, bool inc, NodeDataManager* nodeDataManager){
    for (auto item: {firstItem, secondItem}) {
        if (item == -1) {
            if (item == secondItem) itemset.free();
            continue;
        }
        Array<Item> itemset1 = addItem(itemset, item);
        if (item == secondItem) itemset.free();
        updateItemsetLoad(itemset1, inc);

        auto* node = (HashNode*)get(itemset1);

        if ( node && ((FND)node->data)->left && ((FND)node->data)->right ){
            Item nextFirstItem = item(((FND)node->data)->test, 0);
            Item nextSecondItem = item(((FND)node->data)->test, 1);
            updateSubTreeLoad( itemset1, nextFirstItem, nextSecondItem, inc );
        }
        else if (item == secondItem) itemset1.free();
    }
}*/

/*void Cache_Hash::updateItemsetLoad ( Array<Item> itemset, bool inc ){
    if (store[itemset.size - 1].find(itemset) != store[itemset.size - 1].end() && store[itemset.size - 1][itemset]){
        if (inc) store[itemset.size - 1][itemset]->count_opti_path++;
        else store[itemset.size - 1][itemset]->count_opti_path--;
    }
}*/

/*void Cache_Hash::wipe(Node* node1) {
    for (int i = 0; i < store.size; ++i) {
        for (auto itr = store[i].begin(); itr != store[i].end(); ++itr) {
            if (itr->second->data && itr->second->count_opti_path == 0) store[i].erase(itr->first);
            //else if (itr->second->count_opti_path < 0) for(;;) cout << "g";
        }
    }
}*/

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = itemset[0] + 2 + 0x9e3779b9;
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= itemset[i] + 2 + 0x9e3779b9 + (val<<6) + (val>>2);
        else val ^= 1 + 0x9e3779b9 + (val<<6) + (val>>2);
    }
    return val % maxcachesize;
}*/

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = ((itemset[0] + 2) * pow(37, maxlength-1)) + 0x9e3779b9;
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= ((int)((itemset[i] + 2) * pow(37, maxlength-i-1))) + 0x9e3779b9 + (val<<6) + (val>>2);
        else val ^= ((int)((1) * pow(37, maxlength-i-1))) + 0x9e3779b9 + (val<<6) + (val>>2);
    }
    return val % maxcachesize;
}*/

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = (itemset[0] + 2) * pow(37, maxlength-1);
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= (int)((itemset[i] + 2) * pow(37, maxlength-i-1));
        else val ^= (int)((1) * pow(37, maxlength-i-1));
    }
    return val % maxcachesize;
}*/

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = itemset[0] + 2;
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= itemset[i] + 2;
        else val ^= 1;
    }
    return val % maxcachesize;
}

void Cache_Hash::remove(int hashcode){
    delete bucket[hashcode];
    bucket[hashcode] = nullptr;
}*/

/*pair<Node *, bool> Cache_Hash::insert(Array<Item> itemset, NodeDataManager* nodeDataManager) {
//    cout << "insert" << endl;
    if (itemset.size == 0) {
        root->data = nodeDataManager->initData();
        return {root, true};
    }
    int in_hashcode = gethashcode(itemset);
    if (bucket[in_hashcode]) {
        if (bucket[in_hashcode]->itemset == itemset) return {bucket[in_hashcode], false};
//        if (itemset.size == 1 && itemset[0] == 0) cout << "\t\t\t\t\t\tREM 00000000" << endl;
//        if (in_hashcode == 2) { cout << "\t\t\t\t\t\tREM 00000000" << endl;
//            printItemset(itemset); }
        remove(in_hashcode);
    } else cachesize++;
    Array<Item> copy_itemset;
    copy_itemset.duplicate(itemset); // like copy constructor
    HashNode* node = new HashNode(copy_itemset);
    bucket[in_hashcode] = node;
    node->data = nodeDataManager->initData();
    return {node, true};
}*/
