//
// Created by Gael Aglin on 17/04/2021.
//

#include "solution_Cover.h"

Solution_Cover::Solution_Cover(Search_base *searcher) : Solution(searcher) {
    tree = new Cover_Tree;
}

Solution_Cover::~Solution_Cover() {
    delete tree;
}

Tree* Solution_Cover::getTree() {
    int depth;
    if (searcher->cache->root->data->size == 0 || (searcher->cache->root->data->size == 1 && floatEqual(searcher->cache->root->data->error, FLT_MAX))) {
        tree->expression = "(No such tree)";
        tree->timeout = searcher->timeLimitReached;
    }
    else {
        tree->expression = "";
        depth = printResult(searcher->cache->root->data, 1);
        tree->expression += "}";
        tree->size = searcher->cache->root->data->size;
        tree->depth = depth - 1;
        tree->trainingError = searcher->cache->root->data->error;
        tree->accuracy = 1 - tree->trainingError / float(searcher->nodeDataManager->cover->dm->getNTransactions());
        tree->timeout = searcher->timeLimitReached;
    }
    return (Tree*) tree;
}

int Solution_Cover::printResult(NodeData *data, int depth) {
    if ( ((CoverNodeData*)data)->left == nullptr) { // leaf
        if (searcher->nodeDataManager->tids_error_callback) tree->expression += R"({"value": "undefined", "error": )" + std::to_string(data->error);
        else tree->expression += "{\"value\": " + std::to_string(data->test) + ", \"error\": " + custom_to_str(data->error);
        return depth;
    }
    else {
        tree->expression += "{\"feat\": " + std::to_string((int)data->test) + ", \"left\": ";

        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        int left_depth = printResult( ((CoverNodeData*)data)->right->data, depth + 1);
        tree->expression += "}, \"right\": ";
        int right_depth = printResult( ((CoverNodeData*)data)->left->data, depth + 1);
        tree->expression += "}";
        return max(left_depth, right_depth);
    }
}