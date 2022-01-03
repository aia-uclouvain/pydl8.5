//
// Created by Gael Aglin on 26/09/2020.
//
#ifndef DEPTH_TWO_COMPUTER_H
#define DEPTH_TWO_COMPUTER_H

#include "rCover.h"
#include "cache.h"
#include "solution.h"
#include "nodeDataManagerFreq.h"
#include "rCoverFreq.h"
#include <chrono>
#include <utility>
#include <memory>

using namespace std::chrono;

class Search_base;

Error computeDepthTwo(RCover*, Error, Attributes &, Attribute, Itemset &, Node*, NodeDataManager*, Error, Cache*, Search_base*, bool = false);

struct TreeTwo{
    Freq_NodeData* root_data;

    TreeTwo(){
        root_data = new Freq_NodeData();
    }

    void replaceTree(unique_ptr<TreeTwo> cpy){
//        free();
        root_data = cpy->root_data;
    }

    void free() const {
        if (root_data == nullptr) return;

        if (root_data->left != nullptr){
            if (root_data->left->data != nullptr) delete root_data->left->data->left;
            if (root_data->left->data != nullptr) delete root_data->left->data->right;
            delete root_data->left;
        }

        if (root_data->right != nullptr){
            if (root_data->right->data != nullptr) delete root_data->right->data->left;
            if (root_data->right->data != nullptr) delete root_data->right->data->right;
            delete root_data->right;
        }

        delete root_data;
    }


    ~TreeTwo(){
        free();
    }
};

#endif //DEPTH_TWO_COMPUTER_H
