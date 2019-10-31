#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <unistd.h>
#include "globals.h"
#include "data.h"
#include "dataContinuous.h"
#include "dataBinary.h"
#include "lcm_pruned.h"
#include "query_totalfreq.h"
#include "experror.h"
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string.h>

using namespace std;

bool noTree = true;
bool verbose = false;

int main ( int argc, char *argv[] ) {
	if(argc < 3) {
		cerr << "usage: " << argv[0] << " [-d max] [-s min] [-v] [-i] [-I] [-l] [-n] [-e] [-T] [-t time] datafile " << endl;
        cerr << "-d max: specify maximum depth" <<endl;
        cerr << "-s min: specify minimum support" <<endl;
		cerr << "-i: visit items with high information gain first" <<endl;
		cerr << "-I: visit items with low information gain first" <<endl;
		cerr << "-l: repeat the ordering at each level of the search" <<endl;
		cerr << "-t time: set time limit in seconds" <<endl;
        cerr << "-n: used for continuous dataset" <<endl;
        cerr << "-e: binarize continuous dataset and export it without calculation" << endl;
        cerr << "-T: do not store NO_TREE solutions in the cache" << endl;
        cerr << "-v: verbose" << endl;

		return 1;
	}
	else {
		clock_t t = clock ();

		cerr << "DL8 - Decision Trees from Concept Lattices" << endl;
		cerr << "==========================================" << endl;
		int option;
		Trie *trie = new Trie;
		Query *query = NULL;
		Depth maxdepth = NO_DEPTH;
		bool half = false, j48 = false, infoGain = false, infoAsc = false, allDepths = false, continuous = false, save = false;
		float confc45 = -1.0, confj48;
		int minsup = 1, timeLimit = -1;


		while ( (option = getopt ( argc, argv, "IilnevTd:s:t:" )) != -1 ) {
			switch ( option ) {
				case 'd':
					maxdepth = atoi ( optarg );
					break;
				case 's':
					minsup = atoi ( optarg );
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
					timeLimit = atoi ( optarg );
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

		Data* data;

		if (continuous){
			//cout << "continuous" << endl;
			data = new DataContinuous(save);
		}
		else{
			//cout << "binary" << endl;
			data = new DataBinary;
		}

		data->read ( argv[optind] );
		if (save)
			return 0;

		//create error object and initialize it in the next
		ExpError *experror;



		if ( half ){
			experror = new ExpError_Half;
		}
		else
		if ( j48 ) {
			experror = new ExpError_J48 ( confj48 );
		}
		else
		if ( confc45 == -1.0 ){
			experror = new ExpError_Zero;
		}
		else{
			experror = new ExpError_C45 ( confc45 );
		}

		//query is the object which will answer query about itemset
		//query = new Query_Percentage ( trie, &data, experror );
		query = new Query_TotalFreq ( trie, data, experror, timeLimit, continuous );


		query->maxdepth = maxdepth;
		query->minsup = minsup;

		cout << "\nTrainingDistribution: ";
		forEachClass ( i )
			cout << data->getSupports() [ i ] << " ";
		cout << endl;
		cout << "(nItems, nTransactions) : ( " << data->getNAttributes()*2 << ", " << data->getNTransactions() << " )" << endl;

		LcmPruned lcmPruned( data, query, trie, infoGain, infoAsc, allDepths );
		lcmPruned.run();


		query->printResult ( data );

		cout << "LatticeSize: " << lcmPruned.closedsize << endl;


		cout << "RunTime: " << ( clock () - t ) / (float) CLOCKS_PER_SEC << endl;
	}
	return EXIT_SUCCESS;
}
