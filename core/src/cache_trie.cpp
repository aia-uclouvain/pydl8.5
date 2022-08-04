#include "cache_trie.h"
#include "logger.h"
#include "nodeDataManager_Trie.h"

using namespace std;

bool lessTrieEdge(const TrieEdge edge, const Item item) {
    return edge.item < item;
}

bool lessEdgeItem(const TrieEdge edge1, const TrieEdge edge2) {
    return edge1.item < edge2.item;
}

bool sortDecOrder(TrieNode * &node1, TrieNode * &node2) {
    // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->is_used and not node2->is_used) return true;
    if (not node1->is_used and node2->is_used) return false; // same for the node2
    // sort from highest subnodes to lowest ones. It will be looped for end (lowest) to start (highest)
    return node1->n_subnodes > node2->n_subnodes;
}

bool sortReuseDecOrder(TrieNode * &node1, TrieNode * &node2) {
    // place node1 to left (high value) when it belongs to a potential optimal path
    if (node1->is_used && not node2->is_used) return true;
    if (not node1->is_used && node2->is_used) return false; // same for the node2

    // sort from highest reuse to lowest ones. It will be looped for end (lowest) to start (highest)
    if (node1->n_reuse != node2->n_reuse) return node1->n_reuse > node2->n_reuse;

    // in case reuse is equal for node1 and node2 (itemset with more than one non-existing item), compare depths
    // e.g. when AB exist and ABCD needs to be added, then ABC and ABCD have the same reuse number
    // in this case, loop in depths from bigger to lower, so sorted in increasing order
    return node1->depth < node2->depth;
}

Cache_Trie::Cache_Trie(Depth maxdepth, WipeType wipe_type, int maxcachesize, float wipe_factor) : Cache(maxdepth, wipe_type, maxcachesize), wipe_factor( wipe_factor) {
    root = new TrieNode;
    if (maxcachesize > NO_CACHE_LIMIT) deletion_queue.reserve(maxcachesize - 1);
}

// look for itemset in the trie from root. Return null if not exist and the node of the last item if it exists
Node *Cache_Trie::get(const Itemset &itemset) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    for (const auto &item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != item) return nullptr; // item not found so itemset not found
        else cur_node = geqEdge_it->subtrie;
    }
    return cur_node;
}

// set all subpaths to be useful and return the final path. Return null if it does not exist.
TrieNode *Cache_Trie::getandSet(const Itemset &itemset) {
    auto *cur_node = (TrieNode *) root;
    vector<TrieEdge>::iterator geqEdge_it;
    for (const auto &item: itemset) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), item, lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() || geqEdge_it->item != item) return nullptr; // item not found so itemset not found
        else {
            cur_node = geqEdge_it->subtrie;
            cur_node->is_used = true;
        }
    }
    return cur_node;
}

int Cache_Trie::getCacheSize() {
    if (maxcachesize == NO_CACHE_LIMIT) return cachesize;
    else return deletion_queue.size() + 1;
}

// classic top down
TrieNode *Cache_Trie::addNonExistingItemsetPart(Itemset &itemset, int pos, vector<TrieEdge>::iterator &geqEdge_it, TrieNode *parent_node) {
    TrieNode *child_node;
    for (int i = pos; i < itemset.size(); ++i) {
        child_node = new TrieNode();

        if (maxcachesize > NO_CACHE_LIMIT) {
            deletion_queue.push_back(child_node);
            child_node->trie_parent = parent_node;
        }

        TrieEdge newedge{itemset[i], child_node};
        if (i == pos) parent_node->edges.insert(geqEdge_it, newedge);
        else parent_node->edges.push_back(newedge); // new node added so add the edge without checking its place

        // to prevent the vector containing edges to children to reserve useless space
        /* if (parent_node->edges.size() == parent_node->edges.capacity()) parent_node->edges.reserve(parent_node->edges.size() + 1);
        if (i == pos) {
            geqEdge_it = lower_bound(parent_node->edges.begin(), parent_node->edges.end(), itemset[i], lessTrieEdge);
            parent_node->edges.insert(geqEdge_it, newedge);
        }
        else {
            parent_node->edges.push_back(newedge); // new node added so add the edge without checking its place
        } */

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

    vector<TrieEdge>::iterator geqEdge_it;
    for (int i = 0; i < itemset.size(); ++i) {
        geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
        if (geqEdge_it == cur_node->edges.end() or geqEdge_it->item != itemset[i]) { // the item does not exist
            if (getCacheSize() + itemset.size() - i > maxcachesize && maxcachesize > NO_CACHE_LIMIT) {
                wipe();
                geqEdge_it = lower_bound(cur_node->edges.begin(), cur_node->edges.end(), itemset[i], lessTrieEdge);
            }
            // create path representing the part of the itemset not yet present in the trie.
            TrieNode *last_inserted_node = addNonExistingItemsetPart(itemset, i, geqEdge_it, cur_node);
            return {last_inserted_node, true};
        } else {
            if (i == 0) cur_node->n_reuse++; // root node
            cur_node = geqEdge_it->subtrie;
            cur_node->n_reuse++;
            cur_node->depth = i + 1;
        }
    }

    if (cur_node->data == nullptr) return {cur_node, true};
    else return {cur_node, false};
}

void Cache_Trie::setOptimalNodes(TrieNode *node, const Itemset& itemset) {
    if (node != nullptr and node->data != nullptr and node->data->test >= 0) {
        Itemset left_itemset = addItem(itemset, item(node->data->test, NEG_ITEM));
        auto *left_node = (TrieNode *) getandSet(left_itemset);
        setOptimalNodes(left_node, left_itemset);

        Itemset right_itemset = addItem(itemset, item(node->data->test, POS_ITEM));
        auto *right_node = (TrieNode *) getandSet(right_itemset);
        setOptimalNodes(right_node, right_itemset);

    }
}

void Cache_Trie::setUsingNodes(TrieNode *node, const Itemset &itemset) {
    if (node and node->data and ((TrieNodeData*)node->data)->curr_test != -1) {
        Itemset itemset1 = addItem(itemset, item(((TrieNodeData*)node->data)->curr_test, NEG_ITEM));
        auto node1 = (TrieNode *) getandSet(itemset1);
        setUsingNodes(node1, itemset1);

        Itemset itemset2 = addItem(itemset, item(((TrieNodeData*)node->data)->curr_test, POS_ITEM));
        auto node2 = (TrieNode *) getandSet(itemset2);
        setUsingNodes(node2, itemset2);
    }
}

int Cache_Trie::computeSubNodes(TrieNode *node) {
    node->n_subnodes = 0;
    if (node->edges.empty()) { return 0; }
    for (auto &edge: node->edges) node->n_subnodes += 1 + computeSubNodes(edge.subtrie);
    return node->n_subnodes;
}

void Cache_Trie::wipe() {
    int n_del = (int) (maxcachesize * wipe_factor);
    setOptimalNodes((TrieNode *) root, Itemset());
    Itemset itemset;
    setUsingNodes((TrieNode *) root, itemset);

    // sort the nodes in the cache based on the heuristic used to remove them
    switch (wipe_type) {
        case Subnodes:
            computeSubNodes((TrieNode *) root);
            sort(deletion_queue.begin(), deletion_queue.end(), sortDecOrder);
            break;
        case Reuses:
            sort(deletion_queue.begin(), deletion_queue.end(), sortReuseDecOrder);
            break;
        default: // All. Since the cache structure is a trie, we ensure that the nodes are removed in  bottom-up fashion
            computeSubNodes((TrieNode *) root);
            sort(deletion_queue.begin(), deletion_queue.end(), sortDecOrder);
            n_del = deletion_queue.size();
    }

    int counter = 0;
    for (auto it = deletion_queue.rbegin(); it != deletion_queue.rend(); it++) {
        // the node to remove
        TrieNode *node_del = *it;
        // stop condition
        if (counter == n_del or node_del->is_used) break;

        // remove the edge bringing to the node to remove
        node_del->trie_parent->edges.erase(
                find_if(node_del->trie_parent->edges.begin(), node_del->trie_parent->edges.end(),
                        [node_del](const TrieEdge &look) {
                    return look.subtrie == node_del;
                }));

        // remove the node
        delete node_del;
        // remove its pointer from the deletion queue
        deletion_queue.pop_back();
        counter++;
    }

    // remove the using nodes tag for the next call to wipe
    for (auto &node : deletion_queue) {
        if (node->is_used) node->is_used = false;
        else break;
    }
}
