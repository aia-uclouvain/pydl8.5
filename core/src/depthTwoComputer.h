//
// Created by Gael Aglin on 15/01/2022.
//

#ifndef DL85_DEPTHTWOCOMPUTER_H
#define DL85_DEPTHTWOCOMPUTER_H

#include "rCover.h"
#include "cache_hash_cover.h"
#include "cache_trie.h"
#include "nodeDataManager_Cover.h"
#include "nodeDataManager_Trie.h"
#include "rCoverFreq.h"
#include "depthTwoNodeData.h"
#include <memory>

class Search_base;

Error computeDepthTwo(RCover*, Error, Attributes &, Attribute, const Itemset &, Node*, NodeDataManager*, Error, Cache*, Search_base*, bool = false);



struct TreeTwo{
    DepthTwo_NodeData* root_data;

    TreeTwo(){
        root_data = new DepthTwo_NodeData();
    }

    void replaceTree(unique_ptr<TreeTwo> cpy){
        root_data = cpy->root_data;
    }

    void free() const {
        if (root_data == nullptr) return;

        if (root_data->left != nullptr){
            delete root_data->left->left;
            delete root_data->left->right;
            delete root_data->left;
        }

        if (root_data->right != nullptr){
            delete root_data->right->left;
            delete root_data->right->right;
            delete root_data->right;
        }

        delete root_data;
    }


    ~TreeTwo(){
        free();
    }
};

#endif //DL85_DEPTHTWOCOMPUTER_H
