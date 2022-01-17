//
// Created by Gael Aglin on 14/01/2022.
//

#ifndef DL85_NODE_H
#define DL85_NODE_H

/*This struct is used to represent a node in the tree search algorithm*/
struct Node {
//    NodeData *data; // data is the information kept by a node during the tree search
    bool is_used;
    Node() {
//        data = nullptr;
        is_used = false;
    }
    virtual ~Node() {
//        delete data;
    }
};

#endif //DL85_NODE_H
