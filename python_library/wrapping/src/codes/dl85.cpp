#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

//#include <unistd.h>
//#include <getopt.h>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string.h>
#include <lcm_iterative.h>
#include "globals.h"
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

string search(Supports supports,
              Transaction ntransactions,
              Attribute nattributes,
              Class nclasses,
              Bool *data,
              Class *target,
              float maxError,
              bool stopAfterError,
              bool iterative,
              int maxdepth,
              int minsup,
              bool infoGain,
              bool infoAsc,
              bool repeatSort,
              int timeLimit,
              map<int, pair<int,int>>* continuousMap,
              bool save,
              bool nps,
              bool verbose) {


    /*cout << "argc = " << argc << endl;
    cout << "argv = " ;
    for (int j = 0; j < argc; ++j) {
        cout << argv[j] << " — ";
    }
    cout << endl;*/
    string out = "";
/*
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " [-d max] [-s min] [-v] [-i] [-I] [-l] [-n] [-e] [-T] [-t time] datafile "
             << endl;
        cerr << "-d max: specify maximum depth" << endl;
        cerr << "-s min: specify minimum support" << endl;
        cerr << "-i: visit items with high information gain first" << endl;
        cerr << "-I: visit items with low information gain first" << endl;
        cerr << "-l: repeat the ordering at each level of the search" << endl;
        cerr << "-t time: set time limit in seconds" << endl;
        cerr << "-n: used for continuous dataset" << endl;
        cerr << "-e: binarize continuous dataset and export it without calculation" << endl;
        cerr << "-T: do not store NO_TREE solutions in the cache" << endl;
        cerr << "-v: verbose" << endl;

        return "no work";
    } else {*/
    clock_t t = clock();

    //cerr << "DL8 - Decision Trees from Concept Lattices" << endl;
    //cerr << "==========================================" << endl;
    int option;
    Trie *trie = new Trie;
    Query *query = NULL;
//        Depth maxdepth = NO_DEPTH;
//        bool half = false, j48 = false, infoGain = false, infoAsc = false, allDepths = false, continuous = false, save = false;
//    float confc45 = -1.0, confj48;
//    int minsup = 1, timeLimit = -1;


    /*while ((option = getopt(argc, argv, "IilnevTd:s:t:")) != -1) {
        switch (option) {
            case 'd':
                //cout << 'd' << endl;
                maxdepth = atoi(optarg);
                break;
            case 's':
                //cout << 's' << endl;
                minsup = atoi(optarg);
                break;
            case 'i':
                infoGain = true;
                infoAsc = false;
                break;
            case 'I':
                infoGain = true;
                infoAsc = true;
                break;
            case 'l':
                allDepths = true;
                break;
            case 't':
                //cout << 't' << endl;
                timeLimit = atoi(optarg);
                break;
            case 'T':
                noTree = false;
                break;
            case 'n':
                continuous = true;
                break;
            case 'e':
                continuous = true;
                save = true;
                break;
            case 'v':
                verbose = true;
                break;
        }
    }
    optind = 1;*/

    Data *dataReader;
    dataReader = new DataBinaryPython(supports, ntransactions, nattributes, nclasses, data, target);

    /*if (continuous){
        //cout << "continuous" << endl;
        data = new DataContinuous(save);
    }
    else{
        //cout << "binary" << endl;
        data = new DataBinary;
    }*/

    /*cout << "optind = " << optind << endl;
    cout << "dataset = " << argv[optind] << endl;
    dataReader->read ( argv[optind] );*/
    if (save)
        return 0;

    //create error object and initialize it in the next
    ExpError *experror;
    experror = new ExpError_Zero;


    /*if (half) {
        experror = new ExpError_Half;
    } else if (j48) {
        experror = new ExpError_J48(confj48);
    } else if (confc45 == -1.0) {
        experror = new ExpError_Zero;
    } else {
        experror = new ExpError_C45(confc45);
    }*/

    //query is the object which will answer query about itemset
    //query = new Query_Percentage ( trie, &data, experror );
    query = nullptr;

    if (maxError < 0)
        query = new Query_TotalFreq(trie, dataReader, experror, timeLimit, continuousMap);
    else
        query = new Query_TotalFreq(trie, dataReader, experror, timeLimit, continuousMap, maxError, stopAfterError);


    query->maxdepth = maxdepth;
    query->minsup = minsup;

    out = "TrainingDistribution: ";
    forEachClass (i) out += std::to_string(dataReader->getSupports()[i]) + " ";
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

    //lcm.run();
    //LcmPruned lcmPruned( dataReader, query, trie, infoGain, infoAsc, allDepths );
    //lcmPruned.run();


    out = query->printResult(dataReader);

    //cout << out;
    //cout << "LatticeSize: " << lcmPruned.closedsize << endl;
    //out += "LatticeSize: " + std::to_string(lcmPruned.closedsize) + "\n";// << endl;

    if (iterative)
        out += "LatticeSize: " + std::to_string(((LcmIterative *) lcm)->closedsize) + "\n";// << endl;
    else
        out += "LatticeSize: " + std::to_string(((LcmPruned *) lcm)->closedsize) + "\n";// << endl;


    //cout << "RunTime: " << ( clock () - t ) / (float) CLOCKS_PER_SEC << endl;
    out += "RunTime: " + std::to_string((clock() - t) / (float) CLOCKS_PER_SEC);// << endl;
    //}
    return out;
}
