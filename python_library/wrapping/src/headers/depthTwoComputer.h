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

using namespace std::chrono;

TrieNode* computeDepthTwo(RCover*, Error, Array<Attribute>, Attribute, Array<Item>, TrieNode*, Query*, Error, Trie*);

struct TreeTwo{
    QueryData_Best* root_data;
    QueryData_Best* left_data;
    QueryData_Best* right_data;
    QueryData_Best* left1_data;
    QueryData_Best* left2_data;
    QueryData_Best* right1_data;
    QueryData_Best* right2_data;

    TreeTwo(){
        root_data = new QueryData_Best();
        left_data = new QueryData_Best();
        right_data = new QueryData_Best();
        left1_data = new QueryData_Best();
        left2_data = new QueryData_Best();
        right1_data = new QueryData_Best();
        right2_data = new QueryData_Best();
    }

    TreeTwo(const TreeTwo& cpy){
        root_data = cpy.root_data;
        left_data = cpy.left_data;
        right_data = cpy.right_data;
        left1_data = cpy.left1_data;
        left2_data = cpy.left2_data;
        right1_data = cpy.right1_data;
        right2_data = cpy.right2_data;
    }
};

#endif //DL85_DEPTHTWOCOMPUTER_H
