#include "query_best.h"
#include <iostream>

using namespace std;

Query_Best::Query_Best(Support minsup,
                       Depth maxdepth,
                       Trie* trie,
                       DataManager *data,
                       int timeLimit,
                       function<vector<float>(RCover *)> *tids_error_class_callback,
                       function<vector<float>(RCover *)> *supports_error_class_callback,
                       function<float(RCover *)> *tids_error_callback,
                       float maxError,
                       bool stopAfterError)
        : Query(minsup,
                maxdepth,
                trie,
                data,
                timeLimit,
                tids_error_class_callback,
                supports_error_class_callback,
                tids_error_callback,
                maxError,
                stopAfterError){
}


Query_Best::~Query_Best() {}


void Query_Best::printResult(Tree *tree) {
    printResult((QueryData_Best *) realroot->data, tree);
}

void Query_Best::printResult(QueryData_Best *data, Tree *tree) {
    int depth;
    if (data->size == 0 || (data->size == 1 && floatEqual(data->error, FLT_MAX))) {
        tree->expression = "(No such tree)";
        if (timeLimitReached) tree->timeout = true;
    }
    else {
        tree->expression = "";
        depth = printResult(data, 1, tree);
        tree->expression += "}";
        tree->size = data->size;
        tree->depth = depth - 1;
        tree->trainingError = data->error;
        tree->accuracy = 1 - tree->trainingError / float(dm->getNTransactions());
        if (timeLimitReached) tree->timeout = true;
    }
}

int Query_Best::printResult(QueryData_Best *data, int depth, Tree *tree) {
    if (!data->left) { // leaf
        if (tids_error_callback) tree->expression += R"({"value": "undefined", "error": )" + custom_to_str(data->error);
        else tree->expression += "{\"value\": " + std::to_string(data->test) + ", \"error\": " + custom_to_str(data->error);
        return depth;
    }
    else {
        tree->expression += "{\"feat\": " + std::to_string(data->test) + ", \"left\": ";

        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        int left_depth = printResult(data->right, depth + 1, tree);
        tree->expression += "}, \"right\": ";
        int right_depth = printResult(data->left, depth + 1, tree);
        tree->expression += "}";
        return max(left_depth, right_depth);
    }
}
