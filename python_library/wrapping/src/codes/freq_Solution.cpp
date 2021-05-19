//
// Created by Gael Aglin on 17/04/2021.
//

#include "freq_Solution.h"
#include "lcm_pruned.h"

Freq_Solution::Freq_Solution(void *searcher, NodeDataManager* nodeDataManager1) : Solution(searcher, nodeDataManager1) {
    tree = new Freq_Tree;
}

Freq_Solution::~Freq_Solution() {
    delete tree;
}

Tree* Freq_Solution::getTree() {
    printResult((Freq_NodeData *) ((LcmPruned*)searcher)->cache->root->data);
    return (Tree*) tree;
}

void Freq_Solution::printResult(Freq_NodeData *data) {
    int depth;
    if (data->size == 0 || (data->size == 1 && floatEqual(data->error, FLT_MAX))) {
        tree->expression = "(No such tree)";
        tree->timeout = ((LcmPruned*)searcher)->timeLimitReached;
    }
    else {
        tree->expression = "";
        depth = printResult(data, 1);
        tree->expression += "}";
        tree->size = data->size;
        tree->depth = depth - 1;
        tree->trainingError = data->error;
        tree->accuracy = 1 - tree->trainingError / float(((LcmPruned*)searcher)->nodeDataManager->cover->dm->getNTransactions());
        tree->timeout = ((LcmPruned*)searcher)->timeLimitReached;
    }
}

int Freq_Solution::printResult(Freq_NodeData *data, int depth) {
    if (!data->left) { // leaf
        if (nodeDataManager->tids_error_callback) tree->expression += R"({"value": "undefined", "error": )" + std::to_string(data->error);
        else tree->expression += "{\"value\": " + std::to_string(data->test) + ", \"error\": " + std::to_string(data->error);
        return depth;
    }
    else {
        if (((LcmPruned*)searcher)->continuous) tree->expression += "{\"feat\": " + ((DataContinuous *) nodeDataManager->cover->dm)->names[data->test] + ", \"left\": ";
        else tree->expression += "{\"feat\": " + std::to_string(data->test) + ", \"left\": ";

        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        int left_depth = printResult(data->right, depth + 1);
        tree->expression += "}, \"right\": ";
        int right_depth = printResult(data->left, depth + 1);
        tree->expression += "}";
        return max(left_depth, right_depth);
    }
}