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
        if (root_data->left || root_data->right){
            if (root_data->left->left || root_data->left->right){
                delete root_data->left->left;
                delete root_data->left->right;
            }
            if (root_data->right->left || root_data->right->right){
                delete root_data->right->left;
                delete root_data->right->right;
            }
            delete root_data->left;
            delete root_data->right;
        }
        delete root_data;
    }


    ~TreeTwo(){
        free();
    }
};

#endif //DEPTH_TWO_COMPUTER_H
