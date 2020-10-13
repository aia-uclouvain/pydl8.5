#include "query_best.h"
#include <iostream>
#include <dataContinuous.h>

using namespace std;

Query_Best::Query_Best(Trie* trie,
                       DataManager *data,
                       ExpError *experror,
                       int timeLimit,
                       bool continuous,
                       function<vector<float>(RCover *)> *tids_error_class_callback,
                       function<vector<float>(RCover *)> *supports_error_class_callback,
                       function<float(RCover *)> *tids_error_callback,
                       float maxError,
                       bool stopAfterError)
        : Query(trie,
                data,
                timeLimit,
                continuous,
                tids_error_class_callback,
                supports_error_class_callback,
                tids_error_callback,
                maxError,
                stopAfterError), experror(experror) {
}


Query_Best::~Query_Best() {
}


void Query_Best::printResult(Tree *tree) {
    printResult((QueryData_Best *) realroot->data, tree);
}

void Query_Best::printResult(QueryData_Best *data, Tree *tree) {
    int depth;
    /*string out = "";
    out += "(nItems, nTransactions) : ( " + std::to_string(data2->getNAttributes()*2) + ", " + std::to_string(data2->getNTransactions()) + " )\n";
    out += "Tree: ";*/
    if (data->size == 0 || (data->size == 1 && data->error == FLT_MAX)) {
        tree->expression = "(No such tree)";
        if (timeLimitReached) tree->timeout = true;
    } else {
        tree->expression = "";
        depth = printResult(data, 1, tree);
        tree->expression += "}";
        tree->size = data->size;
        tree->depth = depth - 1;
        if (boosting) tree->trainingError = getTrainingError(tree->expression);
        else tree->trainingError = data->error;
        tree->accuracy = 1 - tree->trainingError / float(dm->getNTransactions());
        // add first function to predict from python
        /*if (!predict_error_callback) tree->add_accuracy(data2->getNTransactions());
        else tree->add_accuracy(data2->getNTransactions(), predict_error_callback(tree->expression));*/
        //printAccuracy(data2, data, tree);
        if (timeLimitReached) tree->timeout = true;
    }
}

int Query_Best::printResult(QueryData_Best *data, int depth, Tree *tree) {
    if (data->left == NULL) { // leaf
        if (tids_error_callback != nullptr)
            tree->expression += "{\"value\": \"undefined\", \"error\": " + std::to_string(data->error);
        else
            tree->expression +=
                    "{\"value\": " + std::to_string(data->test) + ", \"error\": " + std::to_string(data->error);
        return depth;
    } else {
        if (continuous)
            tree->expression += "{\"feat\": " + ((DataContinuous *) this->dm)->names[data->test] + ", \"left\": ";
        else
            tree->expression += "{\"feat\": " + std::to_string(data->test) + ", \"left\": ";
        int d1 = printResult(data->right, depth + 1, tree);
        // perhaps strange, but we have stored the positive outcome in right, generally, people think otherwise... :-)
        tree->expression += "}, \"right\": ";
        int d2 = printResult(data->left, depth + 1, tree);
        tree->expression += "}";
        return max(d1, d2);
    }
}

/*void Query_Best::printTimeOut(Tree *tree) {
    if (timeLimitReached) tree->timeout = true;// << endl;
}*/

/*bool Query_Best::canimprove(QueryData *left, Error ub) {
    return ((QueryData_Best *) left)->error < ub;
}*/

/*bool Query_Best::canSkip(QueryData *actualBest) {
    return ((QueryData_Best *) actualBest)->error == ((QueryData_Best *) actualBest)->lowerBound;
}*/
