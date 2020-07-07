#include "lcm_pruned.h"
#include "query_best.h" // if cannot link is specified, we need a clustering problem!!!
#include "logger.h"
#include <iostream>
#include <limits.h>
#include <cassert>
#include <cmath>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include "dataContinuous.h"


struct Hash {
    size_t operator()(const int &vec) const {
        return vec;
    }
};

LcmPruned::LcmPruned(DataManager *dataReader, Query *query, Trie *trie, bool infoGain, bool infoAsc, bool allDepths) :
        dataReader(dataReader), query(query), trie(trie), infoGain(infoGain), infoAsc(infoAsc), allDepths(allDepths) {
}

LcmPruned::~LcmPruned() {
}

TrieNode *LcmPruned::recurse(Array<Item> itemset_,
                             Item added,
                             Array<Attribute> attributes_to_visit,
                             RCover *cover,
                             Depth depth,
                             float ub) {
    //lowerbound est la petite valeur possible
    //upperbound est la grande valeur inatteignable

    if (query->timeLimit > 0) {
        float runtime = (clock() - query->startTime) / (float) CLOCKS_PER_SEC;
        if (runtime >= query->timeLimit)
            query->timeLimitReached = true;
    }

    Array<Item> itemset;

    if (added != NO_ITEM) {
        itemset.alloc(itemset_.size + 1);
        addItem(itemset_, added, itemset);
    } else {
        itemset.size = 0;
        itemset.elts = nullptr;
    }

    Logger::showMessage("\nitemset avant ajout : ");
    printItemset(itemset_);
    Logger::showMessageAndReturn("Item à ajouter : ", added);
    Logger::showMessage("itemset après ajout : ");
    printItemset(itemset);

    //insert the node in the cache or get its info if it already exists
    TrieNode *node = trie->insert(itemset);

    if (node->data) {//node already exists
        Logger::showMessageAndReturn("le noeud exists");

        Error leafError = ((QueryData_Best *) node->data)->leafError;
        Error *nodeError = &(((QueryData_Best *) node->data)->error);
        Error *lb = &(((QueryData_Best *) node->data)->lowerBound);

        if (*nodeError < FLT_MAX) { //if( nodeError != FLT_MAX ) best solution has been already found
            Logger::showMessageAndReturn("la solution existe et vaut : ", *nodeError);
            itemset.free();
            return node;
        }

        if (ub <= *lb) { //solution impossible
            Logger::showMessageAndReturn("Pas de solution possible car ub < lb. lb =", lb, " et ub = ", ub);
            itemset.free();
            return node;
        }

        if (leafError == *lb) { // implicitely, the upper bound constraint is not violated
            Logger::showMessageAndReturn("l'erreur est minimale");
            *nodeError = leafError;
            itemset.free();
            return node;
        }

        if (depth == query->maxdepth || cover->getSupport() < 2 * query->minsup) {
            Logger::showMessageAndReturn("on a atteint la profondeur maximale. ub = ", ub, " et leaf error = ", leafError);

            if (leafError < ub) {
                *nodeError = leafError;
                Logger::showMessageAndReturn("on retourne leaf error = ", leafError);
            } else {
                *nodeError = FLT_MAX;
                if (ub > *lb) *lb = ub;
                Logger::showMessageAndReturn("pas de solution");
            }
            itemset.free();
            return node;
        }
    }

    //there are two cases in which the execution attempt here
    //1- when the node did not exist
    //2- when the node exists without solution and its upper bound is higher than its lower bound

    Array<Attribute> next_attributes;

    if (!node->data) { // case 1 : when the node did not exist

        /*if (query->maxdepth - depth == 2){
            forEach(i, attributes_to_visit){
                if (item_attribute(added) == attributes_to_visit[i] or current_cover->getSupport() < 2 * query->minsup)
                    continue;
                current_cover->intersect(attributes_to_visit[i], false);
                forEach(j, attributes_to_visit){
                    if (item_attribute(added) == attributes_to_visit[j] || j <= i)
                        continue;
                }
                current_cover->backtrack();
            }
        }*/
        Logger::showMessageAndReturn("Nouveau noeud");
        latticesize++;

        //<=================== STEP 1 : Initialize all information about the node ===================>
        node->data = query->initData(cover, query->minsup);
        Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
        Error leafError = ((QueryData_Best *) node->data)->leafError;
        Error *nodeError = &(((QueryData_Best *) node->data)->error);
        Logger::showMessageAndReturn("après initialisation du nouveau noeud. ub = ", ub, " et leaf error = ", leafError);
        //<====================================  END STEP  ==========================================>


        //<====================== STEP 2 : Case in which we cannot split more =======================>
        if (leafError == *lb) {
            //when leaf error equals to lowerbound all solution parameters have already been stored by initData apart from node error
            *nodeError = leafError;
            Logger::showMessageAndReturn("l'erreur est minimale. node error = leaf error = ", *nodeError);
            itemset.free();
            return node;
        }

        if (depth == query->maxdepth || cover->getSupport() < 2 * query->minsup) {
            Logger::showMessageAndReturn("on a atteint la profondeur maximale. parent bound = ", ub, " et leaf error = ", leafError);
            if (leafError < ub) {
                *nodeError = leafError;
                Logger::showMessageAndReturn("on retourne leaf error = ", leafError);
            } else {
                *nodeError = FLT_MAX;
                if (ub > *lb) *lb = ub;
                Logger::showMessageAndReturn("pas de solution");
            }
            itemset.free();
            return node;
        }

        if (query->timeLimitReached) {
            *nodeError = leafError;
            itemset.free();
            return node;
        }
        //<====================================  END STEP  ==========================================>



        //<============================= STEP 3 : determine successors ==============================>
        next_attributes = getSuccessors(attributes_to_visit, cover, added);
        //<====================================  END STEP  ==========================================>

    } else {//case 2 : when the node exists without solution and ub > lb
//        Error lb = ((QueryData_Best *) node->data)->lowerBound;
//        initUb = parent_ub;
//        ((QueryData_Best *) node->data)->initUb = initUb;
        Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
        Error leafError = ((QueryData_Best *) node->data)->leafError;
        Error *nodeError = &(((QueryData_Best *) node->data)->error);
        Logger::showMessageAndReturn("noeud existant sans solution avec nvelle init bound. leaf error = ",
                                     leafError, " last time: error = ",
                                     *nodeError, " and init = ");
//                                     ((QueryData_Best *) node->data)->initUb, " and stored init = ", storedInit);

        if (query->timeLimitReached) {
            if (*nodeError == FLT_MAX) *nodeError = leafError;
            itemset.free();
            return node;
        }

        //<=========================== ONLY STEP : determine successors =============================>
        next_attributes = getSuccessors(attributes_to_visit, cover, added);
        // next_attributes = (QueryData_Best *) node->data)->successors //if successors have been cached
        // Array<pair<bool, Attribute>> no_attributes = getExistingSuccessors(node); //get successors from trie
        /*if (next_attributes.size != no_attributes.size){ //print for debug
            cout << "itemset size : " << itemset.size << endl;
            cout << "getSuccessors: " << next_attributes.size << endl;
            forEach(i, next_attributes)
                cout << next_attributes[i].second << " : " << next_attributes[i].first << ", ";
            cout << endl;

            cout << "getExistingSuccessors: " << no_attributes.size << endl;
            forEach(i, no_attributes)
                cout << no_attributes[i].second << " : " << no_attributes[i].first << ", ";
            cout << endl;
            cout << endl;
        }*/
        //<====================================  END STEP  ==========================================>
    }

    Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
    Error leafError = ((QueryData_Best *) node->data)->leafError;
    Error *nodeError = &(((QueryData_Best *) node->data)->error);

    if (next_attributes.size == 0) {
        Logger::showMessageAndReturn("pas d'enfant.");
        if (leafError < ub) {
            *nodeError = leafError;
            Logger::showMessageAndReturn("on retourne leaf error = ", leafError);
        } else {
            *nodeError = FLT_MAX;
            if (ub > *lb) *lb = ub;
            Logger::showMessageAndReturn("pas de solution");
        }
        Logger::showMessageAndReturn("on replie");
    }
    else {
        Error child_ub = ub;
        bool notree = true;
        forEach (i, next_attributes) {

            cover->intersect(next_attributes[i], false);
            TrieNode *left = recurse(itemset, item(next_attributes[i], 0), next_attributes, cover, depth + 1, child_ub);
            Error leftError = ((QueryData_Best *) left->data)->error;
            cover->backtrack();

            if (query->canimprove(left->data, child_ub)) {
                notree = false;

                float remainUb = child_ub - leftError;
                cover->intersect(next_attributes[i]);
                TrieNode *right = recurse(itemset, item(next_attributes[i], 1), next_attributes, cover, depth + 1, remainUb);
                Error rightError = ((QueryData_Best *) right->data)->error;
                cover->backtrack();

                Error feature_error = leftError + rightError;
                bool hasUpdated = query->updateData(node->data, child_ub, next_attributes[i], left->data, right->data);
                if (hasUpdated) {
                    child_ub = feature_error;
                    Logger::showMessageAndReturn("après cet attribut, node error = ", *nodeError, " et ub = ", child_ub);
                }

                if (query->canSkip(node->data)) {//lowerBound reached
                    Logger::showMessageAndReturn("C'est le meilleur. on break le reste");
                    break; //prune remaining attributes not browsed yet
                }
            }

            if (query->stopAfterError) {
                if (depth == 0 && ub < FLT_MAX) {
                    if (*nodeError < ub)
                        break;
                }
            }
        }
        if (*nodeError == FLT_MAX && ub > *lb) //cache successors if solution not found
             *lb = ub;
    }


    /*if ((QueryData_Best *) node->data)->error == FLT_MAX) //cache successors if solution not found
        (QueryData_Best *) node->data)->successors = next_attributes;
    else{ //free the cache when solution found
        if ((QueryData_Best *) node->data)->successors != nullptr)
        (QueryData_Best *) node->data)->successors = nullptr;
    }*/

    Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", *nodeError);

    next_attributes.free();
    itemset.free();

    return node;
}


void LcmPruned::run() {
    query->setStartTime(clock());

    //array of items representing an itemset
    Array<Item> itemset;
    itemset.size = 0;

    // array of not yet visited attributes. Each attribute is represented as pair
    // the first element of the pair represent whether or not
    Array<Attribute> attributes_to_visit(nattributes, 0);

    int sup[2];
    RCover *cover = new RCover(dataReader);
    for (int attr = 0; attr < nattributes; ++attr) {
        cover->intersect(attr, false);
        sup[0] = cover->getSupport();
        cover->backtrack();

        if (query->is_freq(make_pair(nullptr, sup[0]))) {
            cover->intersect(attr);
            sup[1] = cover->getSupport();
            cover->backtrack();

            if (query->is_freq(make_pair(nullptr, sup[1])))
                attributes_to_visit.push_back(attr);
        }
    }

    float maxError = NO_ERR;
    if (query->maxError > 0)
        maxError = query->maxError;

    query->realroot = recurse(itemset, NO_ITEM, attributes_to_visit, cover, 0, maxError);

    attributes_to_visit.free();
    delete cover;
}


float LcmPruned::informationGain(pair<Supports, Support> notTaken, pair<Supports, Support> taken) {

    int sumSupNotTaken = notTaken.second;
    int sumSupTaken = taken.second;
    int actualDBSize = sumSupNotTaken + sumSupTaken;

    float condEntropy = 0, baseEntropy = 0;
    float priorProbNotTaken = (actualDBSize != 0) ? (float) sumSupNotTaken / actualDBSize : 0;
    float priorProbTaken = (actualDBSize != 0) ? (float) sumSupTaken / actualDBSize : 0;
    float e0 = 0, e1 = 0;

    for (int j = 0; j < dataReader->getNClasses(); ++j) {
        float p = (sumSupNotTaken != 0) ? (float) notTaken.first[j] / sumSupNotTaken : 0;
        float newlog = (p > 0) ? log2(p) : 0;
        e0 += -p * newlog;

        p = (float) taken.first[j] / sumSupTaken;
        newlog = (p > 0) ? log2(p) : 0;
        e1 += -p * newlog;

        p = (float) (notTaken.first[j] + taken.first[j]) / actualDBSize;
        newlog = (p > 0) ? log2(p) : 0;
        baseEntropy += -p * newlog;
    }
    condEntropy = priorProbNotTaken * e0 + priorProbTaken * e1;

    float actualGain = baseEntropy - condEntropy;

    return actualGain; //high error to low error when it will be put in the map. If you want to have the reverse, just return the negative value of the entropy
}


Array<Attribute> LcmPruned::getSuccessors(Array<Attribute> last_freq_attributes, RCover *cover, Item added) {

    std::multimap<float, Attribute> gain;
    Array<Attribute> a_attributes2(last_freq_attributes.size, 0);
    pair<Supports, Support> supports[2];
    map<int, unordered_set<int, Hash >> control;
    map<int, unordered_map<int, pair<int, float>, Hash>> controle;
    bool to_delete = false;

    if (cover->getSupport() < 2 * query->minsup)
        return a_attributes2;

    forEach (i, last_freq_attributes) {
        if (item_attribute (added) == last_freq_attributes[i])
            continue;
        if (query->error_callback != nullptr || query->predictor_error_callback != nullptr) {//slow or predictor
            cover->intersect(last_freq_attributes[i], false);
            supports[0].second = cover->getSupport();
            cover->backtrack();

            if (query->is_freq(supports[0])){
                cover->intersect(last_freq_attributes[i]);
                supports[1].second = cover->getSupport();
                cover->backtrack();
            } else continue;

        } else { // fast or default
            cover->intersect(last_freq_attributes[i], false);
            supports[0] = cover->getSupportPerClass();
            cover->backtrack();

            if (query->is_freq(supports[0])){
                to_delete = true;
                cover->intersect(last_freq_attributes[i]);
                supports[1] = cover->getSupportPerClass();
                cover->backtrack();
            } else {
                deleteSupports(supports[0].first);
                continue;
            }
        }

        if (query->is_freq(supports[1])) {
            if (query->continuous) {//continuous dataset
            } else {
                if (infoGain)
                    gain.insert(std::pair<float, Attribute>(informationGain(supports[0], supports[1]), last_freq_attributes[i]));
                else a_attributes2.push_back(last_freq_attributes[i]);
            }
        }
        if (to_delete) deleteSupports(supports[1].first);
    }

    if (infoGain) {
        if (infoAsc) { //items with low IG first
            multimap<float, int>::iterator it;
            for (it = gain.begin(); it != gain.end(); ++it) {
                a_attributes2.push_back(it->second);
            }
        } else { //items with high IG first
            multimap<float, int>::reverse_iterator it;
            for (it = gain.rbegin(); it != gain.rend(); ++it) {
                a_attributes2.push_back(it->second);
            }
        }
    }
    if (!allDepths)
        infoGain = false;

    return a_attributes2;
}

void LcmPruned::printItemset(Array<Item> itemset) {
    if (verbose) {
        for (int i = 0; i < itemset.size; ++i) {
            cout << itemset[i] << ",";
        }
        cout << endl;
    }
}

Array<Attribute> LcmPruned::getExistingSuccessors(TrieNode *node) {
    Array<Attribute> a_attributes2(node->edges.size(), 0);
    for (TrieEdge edge : node->edges) {
        if (edge.item % 2 == 0)
            a_attributes2.push_back(item_attribute(edge.item));
    }
    return a_attributes2;
}