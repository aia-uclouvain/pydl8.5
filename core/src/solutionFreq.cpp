//
// Created by Gael Aglin on 17/04/2021.
//

#include "solutionFreq.h"
#include "search_cache.h"

SolutionFreq::SolutionFreq(void *searcher, NodeDataManager* nodeDataManager1) : Solution(searcher, nodeDataManager1) {
    tree = new Freq_Tree;
}

SolutionFreq::~SolutionFreq() {
    delete tree;
}

Tree* SolutionFreq::getTree() {
    printResult((Freq_NodeData *) ((Search_cache*)searcher)->cache->root->data);
    return (Tree*) tree;
}

void SolutionFreq::printResult(Freq_NodeData *data) {
    int depth;
    if (data->size == 0 || (data->size == 1 && floatEqual(data->error, FLT_MAX))) {
        tree->expression = "(No such tree)";
        tree->timeout = ((Search_cache*)searcher)->timeLimitReached;
    }
    else {
        tree->expression = "";
        depth = printResult(data, 1);
        tree->expression += "}";
        tree->size = data->size;
        tree->depth = depth - 1;
        tree->trainingError = data->error;
        tree->accuracy = 1 - tree->trainingError / float(((Search_cache*)searcher)->nodeDataManager->cover->dm->getNTransactions());
        tree->timeout = ((Search_cache*)searcher)->timeLimitReached;
    }
}

int SolutionFreq::printResult(Freq_NodeData *data, int depth) {
    if (!data->left) { // leaf
        if (nodeDataManager->tids_error_callback) tree->expression += R"({"value": "undefined", "error": )" + std::to_string(data->error);
        else tree->expression += "{\"value\": " + std::to_string(data->test) + ", \"error\": " + custom_to_str(data->error);
        return depth;
    }
    else {
        tree->expression += "{\"feat\": " + std::to_string((int)data->test) + ", \"left\": ";

        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        int left_depth = printResult((FND)data->right->data, depth + 1);
        tree->expression += "}, \"right\": ";
        int right_depth = printResult((FND)data->left->data, depth + 1);
        tree->expression += "}";
        return max(left_depth, right_depth);
    }
}