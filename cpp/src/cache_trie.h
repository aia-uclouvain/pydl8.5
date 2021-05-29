#ifndef CACHE_TRIE_H
#define CACHE_TRIE_H

#include "cache.h"
#include <vector>
#include "freq_nodedataManager.h"

using namespace std;

struct TrieNode;

struct TrieEdge {
  Item item;
  TrieNode *subtrie;
};

struct TrieNode : Node {
    vector<TrieEdge> edges;
    int load;
    int count_opti_path;
    TrieNode(): load(-1), count_opti_path(1), Node() {}
    ~TrieNode() {for (auto &edge : edges) delete edge.subtrie;}
//    ~TrieNode() {for ( auto i = edges.begin (); i != edges.end (); ++i ) delete i->subtrie;}
    static bool lte(const TrieEdge edge, const Item item) {
        return edge.item < item;
    }
    /*void update(Attribute attr){
        Item i1 = item(attr, 0); //neg item of the best attribute
        Item i2 = item(attr, 1); //pos item of the best attribute
        for (auto &edge: edges){
            if (edge.item == i1 || edge.item == i2){ // the item belongs to the best attribute
                if (edge.subtrie->load == -1) edge.subtrie->load = 1;
                else edge.subtrie->load++;
            }
            // the item does not belong to the best attribute
            else if (edge.subtrie->load == -1) edge.subtrie->load = 0;
        }
    }*/
    void update(vector<Item>& vec_items, vector<Node*>& vec_nodes) {
        Attribute attr = ((Freq_NodeData*)data)->test;
        Item i1 = item(attr, 0); //neg item of the best attribute
        Item i2 = item(attr, 1); //pos item of the best attribute
        Item item;
        TrieNode* node;
        for (int i = 0; i < vec_items.size(); ++i) {
            item = vec_items[i];
            node = (TrieNode*)vec_nodes[i];
            if (item == i1 || item == i2) { // the item belongs to the best attribute
                if (node->load == -1) node->load = 1;
                else node->load++;
            }
            // the item does not belong to the best attribute
            else if (node->load == -1) node->load = 0;
        }
    }

    void updateNode(Attribute attr, Attribute old, bool hasupdated) {
///        cout << "fia" << endl;
///        for (auto &e: edges) {
///            cout << e.item << ", ";
///        }
///        cout << endl;
///        cout << "test" << endl;
        TrieNode* current_node1 = lower_bound(edges.begin(), edges.end(), item(attr, 0), lte)->subtrie;
        TrieNode* current_node2 = lower_bound(edges.begin(), edges.end(), item(attr, 1), lte)->subtrie;
        ///cout << "fiw" << endl;
        TrieNode* old_node1 = nullptr;
        TrieNode* old_node2 = nullptr;
        if (old >= 0){
            old_node1 = lower_bound(edges.begin(), edges.end(), item(old, 0), lte)->subtrie;
            old_node2 = lower_bound(edges.begin(), edges.end(), item(old, 1), lte)->subtrie;
        }
        ///cout << "fi" << endl;
        if (hasupdated){
            if (current_node1->load == -1) current_node1->load = 1; else current_node1->load++;
            if (current_node2->load == -1) current_node2->load = 1; else current_node2->load++;
            if (old_node1) old_node1->load--; if (old_node2) old_node2->load--;
        } else {
            if (current_node1->load == -1) current_node1->load = 0;
            if (current_node2->load == -1) current_node2->load = 0;
        }
    }

    void changeImportance(TrieNode* first, TrieNode* second, Array<Item> itemset, Item firstI, Item secondI, Cache* cache, bool inc=false) {
//        cout << "change importance" << endl;
        TrieNode* nodes[] = {first, second};
        Array<Item> itemsets[] = { addItem(itemset, firstI), addItem(itemset, secondI) };
        for (auto index: {0, 1}) {
            if (!nodes[index]) continue;
//            cout << "index " << index << endl;
//            cout << "errtt" << endl;
//            if (inc) nodes[index]->count_opti_path++;
            if (inc) cache->uncountItemset(itemsets[index], true);
//            else nodes[index]->count_opti_path--;
            else cache->uncountItemset(itemsets[index]);
//            cout << "fghtr" << endl;
//            printItemset(itemsets[index], true);
            if ( ((FND)nodes[index]->data)->left && ((FND)nodes[index]->data)->right ){
//                cout << "ch" << endl;
//                cout << "1" << endl;
                Item firstI_down = item(((FND)nodes[index]->data)->test, 0);
                TrieNode* first_down = (TrieNode*)cache->get(itemsets[index], firstI_down);
//                cout << "2" << endl;
                Item secondI_down = item(((FND)nodes[index]->data)->test, 1);
                TrieNode* second_down = (TrieNode*)cache->get(itemsets[index], secondI_down);
//                cout << "3" << endl;
                changeImportance( first_down, second_down, itemsets[index], firstI_down, secondI_down, cache, inc );
            }
//            cout << "else" << endl;
        }

    }

    void updateImportance(Node* old_first, Node* old_second, Node* new_first, Node* new_second, bool hasUpdated, Array<Item> itemset, Item old_firstI, Item old_secondI, Item new_firstI, Item new_secondI, Cache* cache){
//        cout << "vvv" << endl;
///cout << "update importance ";
        if (hasUpdated){
            ///cout << "new " << old_first << " " << old_second << endl;
            if (old_first && old_second) changeImportance((TrieNode*)old_first, (TrieNode*)old_first, itemset, old_firstI, old_secondI, cache);
        }
        else {
            ///cout << "old" << endl;
            changeImportance((TrieNode*)new_first, (TrieNode*)new_second, itemset, new_firstI, new_secondI, cache);
        }
    }
};

class Cache_Trie : public Cache {
//friend class Query_TotalFreq;
public:
    Cache_Trie(int maxcachesize=0);
    ~Cache_Trie(){};

    int maxcachesize;
    pair<Node *, bool> insert ( Array<Item> itemset, NodeDataManager* );
    Node *get ( Array<Item> itemset, Item item=-1 );
    void uncountItemset(Array<Item> itemset, bool inc=false);
    void printload(TrieNode* node, vector<Item>& itemset);

private:
    bool canwipe = true;
    void wipe(TrieNode* node, vector<Item>& itemset, int &unsure_count, int &inopti_count);
    TrieNode *createTree ( Array<Item> itemset, int pos, TrieNode *&last, NodeDataManager* nodeDataManager );

};

#endif
