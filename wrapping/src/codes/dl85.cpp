#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string.h>
#include <lcm_iterative.h>
#include "data.h"
#include "dataContinuous.h"
#include "dataBinary.h"
#include "dataBinaryPython.h"
#include "lcm_pruned.h"
#include "query_totalfreq.h"
#include "experror.h"
#include "dataManager.h"

//using namespace std;

bool nps = false;
bool verbose = false;

string search(//std::function<float(Array<int>::iterator)> callback,
              Supports supports,
              Transaction ntransactions,
              Attribute nattributes,
              Class nclasses,
              Bool *data,
              Class *target,
              float maxError,
              bool stopAfterError,
              bool iterative,
              function<vector<float>(RCover*)> error_callback,
              function<vector<float>(RCover*)> fast_error_callback,
              function<float(RCover*)> predictor_error_callback,
              bool error_is_null,
              bool fast_error_is_null,
              int maxdepth,
              int minsup,
              bool infoGain,
              bool infoAsc,
              bool repeatSort,
              int timeLimit,
              map<int, pair<int, int>> *continuousMap,
              bool save,
              bool nps_param,
              bool verbose_param,
              bool predict) {

    clock_t t = clock();

    function<vector<float>(RCover*)> *error_callback_pointer = &error_callback;
    function<vector<float>(RCover*)> *fast_error_callback_pointer = &fast_error_callback;
    function<float(RCover*)> *predictor_error_callback_pointer = &predictor_error_callback;

    if (error_is_null)
        error_callback_pointer = nullptr;
    if (fast_error_is_null)
        fast_error_callback_pointer = nullptr;
    if(!predict)
        predictor_error_callback_pointer = nullptr;

    //cout << "print " << fast_error_callback->pyFunction << endl;
    nps = nps_param;
    verbose = verbose_param;
    string out = "";
    Trie *trie = new Trie;
    Query *query = NULL;

    DataManager *dataReader = new DataManager(supports, ntransactions, nattributes, nclasses, data, target);

    if (save)
        return 0;

    //create error object and initialize it in the next
    ExpError *experror;
    experror = new ExpError_Zero;

    if (maxError <= 0)
        query = new Query_TotalFreq(trie, dataReader, experror, timeLimit, continuousMap, error_callback_pointer, fast_error_callback_pointer, predictor_error_callback_pointer);
    else
        query = new Query_TotalFreq(trie, dataReader, experror, timeLimit, continuousMap, error_callback_pointer, fast_error_callback_pointer, predictor_error_callback_pointer, maxError, stopAfterError);

    query->maxdepth = maxdepth;
    query->minsup = minsup;

    out = "TrainingDistribution: ";
    forEachClass(i)
    out += std::to_string(dataReader->getSupports()[i]) + " ";
    out += "\n";
    //out += "(nItems, nTransactions) : ( " << std::to_string(dataReader->getNAttributes()*2) << ", " << std::to_string(dataReader->getNTransactions()) << " )" << endl;

    void *lcm;
    if (iterative) {
        lcm = new LcmIterative(dataReader, query, trie, infoGain, infoAsc, repeatSort);
        ((LcmIterative *) lcm)->run();
    } else {
        lcm = new LcmPruned(dataReader, query, trie, infoGain, infoAsc, repeatSort);
        ((LcmPruned *) lcm)->run();
    }

    out = query->printResult(dataReader);

    if (iterative)
        out += "LatticeSize: " + std::to_string(((LcmIterative *) lcm)->latticesize) + "\n";
    else
        out += "LatticeSize: " + std::to_string(((LcmPruned *) lcm)->latticesize) + "\n";

    out += "RunTime: " + std::to_string((clock() - t) / (float) CLOCKS_PER_SEC);

    return out;
}
