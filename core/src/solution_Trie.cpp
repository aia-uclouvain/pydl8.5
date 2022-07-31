//
// Created by Gael Aglin on 17/04/2021.
//

#include "solution_Trie.h"

Solution_Trie::Solution_Trie(Search_base *searcher) : Solution(searcher) {
    tree = new Trie_Tree;
}

Solution_Trie::~Solution_Trie() {
    delete tree;
}

Tree* Solution_Trie::getTree() {
    int depth;
    if (searcher->cache->root->data->size == 0 || (searcher->cache->root->data->size == 1 && floatEqual(searcher->cache->root->data->error, FLT_MAX))) {
        tree->expression = "(No such tree)";
        tree->timeout = searcher->timeLimitReached;
    }
    else {
        tree->expression = "";
        depth = printResult(searcher->cache->root, 1, Itemset());
        tree->expression += "}";
        tree->size = searcher->cache->root->data->size;
        tree->depth = depth - 1;
        tree->trainingError = searcher->cache->root->data->error;
        tree->accuracy = 1 - tree->trainingError / float(searcher->nodeDataManager->cover->dm->getNTransactions());
        tree->timeout = searcher->timeLimitReached;
    }
    return (Tree*) tree;
}

int Solution_Trie::printResult(Node* node, int depth, const Itemset& itemset) {

    // the variable `test` is used to save the feature to split on branch nodes and classes at leaf nodes
    // to make them distinguishable, the feature value is positive and classes are transformed by f(x) = -(x+1)
    // attention to recover the right value when printing the tree
    if (node->data->test < 0) { // leaf
        if (searcher->nodeDataManager->tids_error_callback) tree->expression += R"({"value": "undefined", "error": )" + std::to_string(node->data->error);
        else tree->expression += "{\"value\": " + std::to_string(-node->data->test - 1) + ", \"error\": " + custom_to_str(node->data->error);
        return depth;
    }
    else {
        tree->expression += "{\"feat\": " + std::to_string((int)node->data->test) + ", \"left\": ";

        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        Itemset itemset_right = addItem(itemset, item(node->data->test, POS_ITEM));
        Node* node_right = searcher->cache->get(itemset_right);
        int left_depth = printResult(node_right, depth + 1, itemset_right);
        tree->expression += "}, \"right\": ";

        Itemset itemset_left = addItem(itemset, item(node->data->test, NEG_ITEM));
        Node* node_left = searcher->cache->get(itemset_left);
        int right_depth = printResult(node_left, depth + 1, itemset_left);
        tree->expression += "}";
        return max(left_depth, right_depth);
    }
}