//
// Created by Gael Aglin on 26/09/2020.
//
#ifndef DL85_DEPTHTWOCOMPUTER_H
#define DL85_DEPTHTWOCOMPUTER_H

#include "rCover.h"
#include "trie.h"
#include "query.h"
#include "query_best.h"
#include <chrono>
#include <utility>
#include <memory>

using namespace std::chrono;

TrieNode* computeDepthTwo(RCover*, Error, Array<Attribute>, Attribute, Array<Item>, TrieNode*, Query*, Error, Trie*);

struct TreeTwo{
    QueryData_Best* root_data;

    TreeTwo(){
        root_data = new QueryData_Best();
    }

    void replaceTree(TreeTwo* cpy){
        free();
        root_data = cpy->root_data;
    }

    void free(){
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
