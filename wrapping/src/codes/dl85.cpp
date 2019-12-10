#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <functional>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string.h>
//#include <lcm_iterative.h>
#include "data.h"
#include "dataContinuous.h"
#include "dataBinary.h"
#include "dataBinaryPython.h"
#include "lcm_pruned.h"
#include "query_totalfreq.h"
#include "experror.h"

using namespace std;

bool nps = false;
bool verbose = false;

string search(//std::function<float(Array<int>::iterator)> callback,
              function<vector<float>(Array<int>*)> error_callback,
              Supports supports,
              Transaction ntransactions,
              Attribute nattributes,
              Class nclasses,
              Bool *data,
              Class *target,
              float maxError,
              bool stopAfterError,
              bool iterative,
              bool user,
              int maxdepth,
              int minsup,
              bool infoGain,
              bool infoAsc,
              bool repeatSort,
              int timeLimit,
              map<int, pair<int, int>> *continuousMap,
              bool save,
              bool nps_param,
              bool verbose_param) {

    /*Array<int> ar(3,3);
    ar[0] = 10;
    ar[1] = 20;
    ar[2] = 30;
    //cout << "computed error is " << endl << callback(ar.begin()) << endl;
    cout << "computed error is " << endl << callback(&ar) << endl;
    ar.free();*/

    nps = nps_param;
    verbose = verbose_param;
    string out = "";
    clock_t t = clock();
    Trie *trie = new Trie;
    Query *query = NULL;


    Data *dataReader;
    dataReader = new DataBinaryPython(supports, ntransactions, nattributes, nclasses, data, target);

    if (save)
        return 0;

    //create error object and initialize it in the next
    ExpError *experror;
    experror = new ExpError_Zero;

    if (maxError < 0)
        query = new Query_TotalFreq(trie, dataReader, experror, timeLimit, continuousMap, &error_callback);
    else
        query = new Query_TotalFreq(trie, dataReader, experror, timeLimit, continuousMap, &error_callback, maxError, stopAfterError);


    query->maxdepth = maxdepth;
    query->minsup = minsup;

    out = "TrainingDistribution: ";
    forEachClass(i)
    out += std::to_string(dataReader->getSupports()[i]) + " ";
    out += "\n";
    //out += "(nItems, nTransactions) : ( " << std::to_string(dataReader->getNAttributes()*2) << ", " << std::to_string(dataReader->getNTransactions()) << " )" << endl;

    void *lcm;

    if (iterative) {
        cout << "it" << endl;
        //lcm = new LcmIterative(dataReader, query, trie, infoGain, infoAsc, repeatSort);
        //((LcmIterative *) lcm)->run();
    } else {
        lcm = new LcmPruned(dataReader, query, trie, infoGain, infoAsc, repeatSort, user);
        ((LcmPruned *) lcm)->run();
    }

    out = query->printResult(dataReader);

    if (iterative)
        cout << "it" << endl;
        //out += "LatticeSize: " + std::to_string(((LcmIterative *) lcm)->closedsize) + "\n";// << endl;
    else
        out += "LatticeSize: " + std::to_string(((LcmPruned *) lcm)->closedsize) + "\n";// << endl;

    out += "RunTime: " + std::to_string((clock() - t) / (float) CLOCKS_PER_SEC);// << endl;

    return out;
}
