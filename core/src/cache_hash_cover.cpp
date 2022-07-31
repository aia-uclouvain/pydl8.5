#include "cache_hash_cover.h"
#include "nodeDataManager_Cover.h"

using namespace std;

Cache_Hash_Cover::Cache_Hash_Cover(Depth maxdepth, WipeType wipe_type, int maxcachesize, float wipe_factor) :
        Cache(maxdepth, wipe_type, maxcachesize), wipe_factor( wipe_factor) {
    root = new HashCoverNode();
    store = new unordered_map<MyCover, HashCoverNode *>[maxdepth];

    // reserve the size needed to keep the iterators
    /*if (this->maxcachesize > NO_CACHE_LIMIT) {
        deletion_queue = vector<pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Depth>>;
        deletion_queue.reserve(maxcachesize);
    }*/

}

pair<Node *, bool> Cache_Hash_Cover::insert(NodeDataManager *nodeDataManager, int depth) {

    if (depth == 0) {
        cachesize++;
        return {root, true};
    } else {
        /*if (maxcachesize > NO_CACHE_LIMIT and getCacheSize() >= maxcachesize) {
            wipe();
        }*/
        auto *node = new HashCoverNode();
        auto info = store[depth-1].insert({MyCover(nodeDataManager->cover), node});
        if (not info.second) { // if node already exists
            delete node;
            info.first->second->n_reuse++;
        }
        else {
            /*if (maxcachesize > NO_CACHE_LIMIT) {
                deletion_queue.push_back(make_pair(&(info.first), depth));
            }*/
        }
        return {info.first->second, info.second};
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

/*bool sortReuseDecOrder(pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Depth> &pair1, pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Depth> &pair2) {
    const auto node1 = (*(pair1.first))->second, node2 = (*(pair2.first))->second;
    if (node1->is_used && not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used && node2->is_used) return false; // same for the node2
    return node1->n_reuse > node2->n_reuse; // in case both nodes are in potential optimal paths or both of not
}

bool sortDecOrder(pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Depth> &pair1, pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Depth> &pair2) {
    const auto node1 = (*(pair1.first))->second, node2 = (*(pair2.first))->second;
    if (node1->is_used && not node2->is_used)
        return true; // place node1 to left (high value) when it belongs to a potential optimal path
    if (not node1->is_used && node2->is_used) return false; // same for the node2
    return node1->n_subnodes > node2->n_subnodes; // in case both nodes are in potential optimal paths or both of not
}

void Cache_Hash_Cover::setOptimalNodes(HashCoverNode *node) {
    if (((CoverNodeData*)node->data)->left != nullptr) {
        ((CoverNodeData*)node->data)->left->is_used = true;
        setOptimalNodes((HashCoverNode *) ((CoverNodeData*)node->data)->left);

        ((HashCoverNode *) ((CoverNodeData*)node->data)->right)->is_used = true;
        setOptimalNodes((HashCoverNode *) ((CoverNodeData*)node->data)->right);
    }
}

void Cache_Hash_Cover::setUsingNodes(HashCoverNode *node) {
    if (node and node->data and ((CoverNodeData*)node->data)->curr_left != nullptr) {
        ((CoverNodeData*)node->data)->curr_left->is_used = true;
        setUsingNodes((HashCoverNode *) ((CoverNodeData*)node->data)->curr_left);
    }

    if (node and node->data and ((CoverNodeData*)node->data)->curr_right != nullptr) {
        ((CoverNodeData*)node->data)->curr_right->is_used = true;
        setUsingNodes((HashCoverNode *) ((CoverNodeData*)node->data)->curr_right);
    }
}

void Cache_Hash_Cover::wipe() {

    int n_del = (int) (maxcachesize * wipe_factor);
    setOptimalNodes((HashCoverNode *) root);
    setUsingNodes((HashCoverNode *) root);

    // sort the nodes in the cache based on the heuristic used to remove them
    sort(deletion_queue.begin(), deletion_queue.end(), sortReuseDecOrder);

     cout << "cachesize before wipe = " << getCacheSize() << endl;
    cout << n_del << " " << maxcachesize << " " << wipe_factor << endl;
    int counter = 0;

    // before removing nodes, sort the deletion from deeper depth to shallower while ensuring that for each
    // depth, the iterators are sorted in order to prevent iterators invalidation. Then remove them from end to start
    for (auto it = deletion_queue.rbegin(); it != deletion_queue.rend(); it++) {
        // the node to remove
        const unordered_map<MyCover, HashCoverNode*>::iterator* node_del = it->first;
        Depth depth_to_del = it->second;
        delete (*node_del)->second;
        store[depth_to_del].erase(*node_del);
        counter++;
        if (counter == n_del or (*node_del)->second->is_used) {
            break;
        }
    }
    // reinitialize the deletion queue otherwise the iterators will be invalidated. for this, clear
    // the deletion queue. Loop in the store and insert at new the nodes and correct iterator values
    // this is not done in the code but it is important to work properly

     cout << "cachesize after wipe = " << getCacheSize() << endl;
}*/

