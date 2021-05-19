//
// Created by Gael Aglin on 26/09/2020.
//
#ifndef DL85_DEPTHTWOCOMPUTER_H
#define DL85_DEPTHTWOCOMPUTER_H

#include "rCover.h"
#include "cache.h"
#include "freq_nodedataManager.h"
#include "solution.h"
#include <chrono>
#include <utility>
#include "rCoverTotalFreq.h"

using namespace std::chrono;

Node* computeDepthTwo(Error, Array<Attribute>, Attribute, Array<Item>, Node*, NodeDataManager*, Error, Cache*);

struct TreeTwo{
    Freq_NodeData* root_data;
    Freq_NodeData* left_data;
    Freq_NodeData* right_data;
    Freq_NodeData* left1_data;
    Freq_NodeData* left2_data;
    Freq_NodeData* right1_data;
    Freq_NodeData* right2_data;

    TreeTwo(){
        root_data = new Freq_NodeData();
        left_data = new Freq_NodeData();
        right_data = new Freq_NodeData();
        left1_data = new Freq_NodeData();
        left2_data = new Freq_NodeData();
        right1_data = new Freq_NodeData();
        right2_data = new Freq_NodeData();
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
