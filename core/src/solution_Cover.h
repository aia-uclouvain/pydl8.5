//
// Created by Gael Aglin on 17/04/2021.
//

#ifndef SOLUTION_COVER_FREQ_H
#define SOLUTION_COVER_FREQ_H

#include "solution.h"
#include "search_cover_cache.h"

/**
 * This structure a decision tree model learnt from input data
 * @param expression - a json string representing the tree
 * @param size - the number of nodes (branches + leaves) in the tree
 * @param depth - the depth of the tree; the length of the longest rule in the tree
 * @param trainingError - the error of the tree on the training set given the used objective function
 * @param latSize - the number of nodes explored before finding the solution. Currently this value is not correct :-(
 * @param searchRt - the time that the search took
 * @param timeout - a boolean variable to represent the fact that the search reached a timeout or not
 */
struct Cover_Tree : Tree {

    string to_str() const override {
        string out = "";
        out += "Tree: " + expression + "\n";
        out += (expression != "(No such tree)") ? "Size: " + to_string(size) + "\n" : "Size: 0\n";
        out += (expression != "(No such tree)") ? "Depth: " + to_string(depth) + "\n" : "Depth: 0\n";
        out += (expression != "(No such tree)") ? "Error: " + custom_to_str(trainingError) + "\n" : "Error: inf\n";
        out += (expression != "(No such tree)") ? "Accuracy: " + custom_to_str(accuracy) + "\n" : "Accuracy: 0\n";
        out += "CacheSize: " + to_string(cacheSize) + "\n";
        out += "RunTime: " + custom_to_str(runtime) + "\n";
        if (timeout) out += "Timeout: True\n";
        else out += "Timeout: False\n";
        return out;
    }

    ~Cover_Tree() override {}
};


class Solution_Cover : public Solution {
public:
    Solution_Cover(Search_base*);

    ~Solution_Cover();

    Tree * getTree();

protected:
    int printResult(NodeData *node_data, int depth);
};

#endif //SOLUTION_COVER_FREQ_H
