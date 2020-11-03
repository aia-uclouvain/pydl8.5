#include "query_best.h"
#include <iostream>
#include <dataContinuous.h>

using namespace std;

Query_Best::Query_Best(Support minsup,
                       Depth maxdepth,
                       Trie* trie,
                       DataManager *data,
                       ExpError *experror,
                       int timeLimit,
                       bool continuous,
                       function<vector<float>(RCover *)> *tids_error_class_callback,
                       function<vector<float>(RCover *)> *supports_error_class_callback,
                       function<float(RCover *)> *tids_error_callback,
                       function<vector<float>()> *example_weight_callback,
                       function<vector<float>(string)> *predict_error_callback,
                       float maxError,
                       bool stopAfterError)
        : Query(minsup,
                maxdepth,
                trie,
                data,
                timeLimit,
                continuous,
                tids_error_class_callback,
                supports_error_class_callback,
                tids_error_callback,
                example_weight_callback,
                predict_error_callback,
                maxError,
                stopAfterError), experror(experror) {
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
        if (tids_error_callback) tree->expression += R"({"value": "undefined", "error": )" + std::to_string(data->error);
        else tree->expression += "{\"value\": " + std::to_string(data->test) + ", \"error\": " + std::to_string(data->error);
        return depth;
    }
    else {
        if (continuous) tree->expression += "{\"feat\": " + ((DataContinuous *) this->dm)->names[data->test] + ", \"left\": ";
        else tree->expression += "{\"feat\": " + std::to_string(data->test) + ", \"left\": ";

        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        int left_depth = printResult(data->right, depth + 1, tree);
        tree->expression += "}, \"right\": ";
        int right_depth = printResult(data->left, depth + 1, tree);
        tree->expression += "}";
        return max(left_depth, right_depth);
    }
}
