//
// Created by Gael Aglin on 15/01/2022.
//

#ifndef DL85_DEPTHTWONODEDATA_H
#define DL85_DEPTHTWONODEDATA_H

#include "nodeDataManager.h"

struct DepthTwo_NodeData : NodeData {

    DepthTwo_NodeData *left, *right;

    DepthTwo_NodeData(): NodeData() {
        left = nullptr;
        right = nullptr;
    }

};

#endif //DL85_DEPTHTWONODEDATA_H
