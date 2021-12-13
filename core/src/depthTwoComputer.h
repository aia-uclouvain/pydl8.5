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

using namespace std::chrono;

class Search_base;

Error computeDepthTwo(RCover*, Error, Attributes &, Attribute, Itemset &, Node*, NodeDataManager*, Error, Cache*, Search_base*, bool = false);

struct TreeTwo{
    Freq_NodeData* root_data;

    TreeTwo(){
        root_data = new Freq_NodeData();
    }

    void replaceTree(TreeTwo* cpy){
        free();
        root_data = cpy->root_data;
    }

    void free(){
        if (root_data->left != nullptr or root_data->right != nullptr){
            if (root_data->left->data->left != nullptr or root_data->left->data->right != nullptr){
                delete root_data->left->data->left;
                delete root_data->left->data->right;
            }
            if (root_data->right->data->left != nullptr or root_data->right->data->right != nullptr){
                delete root_data->right->data->left;
                delete root_data->right->data->right;
            }
            delete root_data->left;
            delete root_data->right;
        }
        delete root_data;
    }


    ~TreeTwo(){
    }
};

#endif //DEPTH_TWO_COMPUTER_H
