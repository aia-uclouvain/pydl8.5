#include "lcm_pruned.h"
#include "query_best.h" // if cannot link is specified, we need a clustering problem!!!
#include "logger.h"
#include <iostream>
#include <limits.h>
#include <cassert>
#include <cmath>
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

TrieNode *LcmPruned::recurse(Array<Item> itemset,
                             Item added,
                             TrieNode *node,
                             Array<Attribute> attributes_to_visit,
                             RCover *cover,
                             Depth depth,
                             float ub,
                             float lbb) {
    //lowerbound est la petite valeur possible
    //upperbound est la grande valeur inatteignable

    if (query->timeLimit > 0) {
        float runtime = (clock() - query->startTime) / (float) CLOCKS_PER_SEC;
        if (runtime >= query->timeLimit)
            query->timeLimitReached = true;
    }

    if (node->data) {//node already exists
        Logger::showMessageAndReturn("le noeud exists");

        Error *nodeError = &(((QueryData_Best *) node->data)->error);
        if (*nodeError < FLT_MAX) { //if( nodeError != FLT_MAX ) best solution has been already found
            Logger::showMessageAndReturn("la solution existe et vaut : ", *nodeError);
            return node;
        }

        Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
        if (lbb > *lb) *lb = lbb;
        if (ub <= *lb) { //solution impossible
            Logger::showMessageAndReturn("Pas de solution possible car ub < lb. lb =", lb, " et ub = ", ub);
            return node;
        }

        Error leafError = ((QueryData_Best *) node->data)->leafError;
        if (leafError == *lb) { // implicitely, the upper bound constraint is not violated
            Logger::showMessageAndReturn("l'erreur est minimale");
            *nodeError = leafError;
            return node;
        }

        if (depth == query->maxdepth || cover->getSupport() < 2 * query->minsup) {
            Logger::showMessageAndReturn("on a atteint la profondeur maximale. ub = ", ub, " et leaf error = ", leafError);
            *nodeError = leafError; //added
            /*if (leafError < ub) {
                *nodeError = leafError;
                Logger::showMessageAndReturn("on retourne leaf error = ", leafError);
            } else {
                if (ub > *lb) *lb = ub;
                Logger::showMessageAndReturn("pas de solution");
            }*/
            return node;
        }
    }

    if (query->maxdepth - depth == 2 && cover->getSupport() >= 2 * query->minsup){
        // lbb = (node->data) ? max(lbb, ((QueryData_Best *) node->data)->lowerBound) : lbb;
        return getdepthtwotrees(cover, ub, attributes_to_visit, added, itemset, node, lbb);
    }
    else {
        //there are two cases in which the execution attempt here
        //1- when the node did not exist
        //2- when the node exists without solution and its upper bound is higher than its lower bound

        Array<Attribute> next_attributes;

        if (!node->data) { // case 1 : when the node did not exist
            Logger::showMessageAndReturn("Nouveau noeud");
            latticesize++;

            //<=================== STEP 1 : Initialize all information about the node ===================>
            node->data = query->initData(cover);
            Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
            if (lbb > *lb) *lb = lbb;
            Error leafError = ((QueryData_Best *) node->data)->leafError;
            Error *nodeError = &(((QueryData_Best *) node->data)->error);
            Logger::showMessageAndReturn("après initialisation du nouveau noeud. ub = ", ub, " et leaf error = ", leafError);
            //<====================================  END STEP  ==========================================>


            //<====================== STEP 2 : Case in which we cannot split more =======================>
            if (ub <= *lb){
                Logger::showMessageAndReturn("impossible car la ub < lb. lb = ", lb, " et ub = ", ub);
                return node;
            }
            if (leafError == *lb) { //when leaf error equals to lowerbound all solution parameters have already been stored by initData apart from node error
                *nodeError = leafError;
                Logger::showMessageAndReturn("l'erreur est minimale. node error = leaf error = ", *nodeError);
                return node;
            }
            if (depth == query->maxdepth || cover->getSupport() < 2 * query->minsup) {
                Logger::showMessageAndReturn("on a atteint la profondeur maximale. parent bound = ", ub, " et leaf error = ", leafError);
                *nodeError = leafError;
                /*if (leafError < ub) {
                    *nodeError = leafError;
                    Logger::showMessageAndReturn("on retourne leaf error = ", leafError);
                } else {
                    if (ub > *lb) *lb = ub;
                    Logger::showMessageAndReturn("pas de solution");
                }*/
                return node;
            }

            if (query->timeLimitReached) {
                *nodeError = leafError;
                return node;
            }
            //<====================================  END STEP  ==========================================>



            //<============================= STEP 3 : determine successors ==============================>
            next_attributes = getSuccessors(attributes_to_visit, cover, added);
            //<====================================  END STEP  ==========================================>

        } else {//case 2 : when the node exists without solution and ub > lb
            Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
            if (lbb > *lb) *lb = lbb;
            Error leafError = ((QueryData_Best *) node->data)->leafError;
            Error *nodeError = &(((QueryData_Best *) node->data)->error);
            Logger::showMessageAndReturn("noeud existant sans solution avec nvelle init bound. leaf error = ",
                                         leafError, " new ub = ", ub);

            if (query->timeLimitReached) {
                if (*nodeError == FLT_MAX) *nodeError = leafError;
                return node;
            }

            //<=========================== ONLY STEP : determine successors =============================>
            // next_attributes = getSuccessors(attributes_to_visit, cover, added, getExistingSuccessors(node));
            next_attributes = getSuccessors(attributes_to_visit, cover, added);
            //<====================================  END STEP  ==========================================>
        }

        Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
        Error leafError = ((QueryData_Best *) node->data)->leafError;
        Error *nodeError = &(((QueryData_Best *) node->data)->error);

        if (next_attributes.size == 0) {
            Logger::showMessageAndReturn("pas d'enfant.");
            *nodeError = leafError;
            /*if (leafError < ub) {
                *nodeError = leafError;
                Logger::showMessageAndReturn("on retourne leaf error = ", leafError);
            } else {
                if (ub > *lb) *lb = ub;
                Logger::showMessageAndReturn("pas de solution");
            }*/
            Logger::showMessageAndReturn("on replie");
        }
        else {
            Error child_ub = ub;
            //Array<bitset<M>*> covers(next_attributes.size * 2, 0);
            //Array<float> errors(next_attributes.size * 2, 0);
            Error minlb = 0;
            bitset<M>* covlb1 = nullptr; Supports sclb1 = nullptr; Supports sflb1 = nullptr; Error errlb1 = 0;
            bitset<M>* covlb2 = nullptr; Supports sclb2 = nullptr; Supports sflb2 = nullptr; Error errlb2 = FLT_MAX;
            bitset<M>* covlb3 = nullptr; Supports sclb3 = nullptr; Supports sflb3 = nullptr; Support suplb = 0; Error errlb3;
            Support parentsup = cover->getSupport();
            forEach (i, next_attributes) {

                /*cover->intersect(next_attributes[i], false);
                Error llb = computeLowerBound(cover, covers, errors);
                cover->backtrack();

                cover->intersect(next_attributes[i]);
                Error rlb = computeLowerBound(cover, covers, errors);
                cover->backtrack();

                TrieNode* left_node;
                TrieNode* right_node;
                if (llb <= rlb){
                    cover->intersect(next_attributes[i], false);
                    Item left_item = item(next_attributes[i], 0);
                    Array<Item> left_itemset = addItem(itemset, left_item);
                    left_node = trie->insert(left_itemset);
                    left_node = recurse(left_itemset, left_item, left_node, next_attributes, cover, depth + 1, child_ub, llb);
                    addInfoForLowerBound(cover, left_node->data, covers, errors);
                    Error leftError = ((QueryData_Best *) left_node->data)->error;
                    left_itemset.free();
                    cover->backtrack();

                    if (query->canimprove(left_node->data, child_ub)) {

                        float remainUb = child_ub - leftError;
                        cover->intersect(next_attributes[i]);
                        Item right_item = item(next_attributes[i], 1);
                        Array<Item> right_itemset = addItem(itemset, right_item);
                        right_node = trie->insert(right_itemset);
                        right_node = recurse(right_itemset, right_item, right_node, next_attributes, cover, depth + 1, remainUb, rlb);
                        addInfoForLowerBound(cover, right_node->data, covers, errors);
                        Error rightError = ((QueryData_Best *) right_node->data)->error;
                        right_itemset.free();
                        cover->backtrack();

                        Error feature_error = leftError + rightError;
                        bool hasUpdated = query->updateData(node->data, child_ub, next_attributes[i], left_node->data, right_node->data);
                        if (hasUpdated) child_ub = feature_error;
                        if (hasUpdated) Logger::showMessageAndReturn("après cet attribut, node error = ", *nodeError, " et ub = ", child_ub);

                        if (query->canSkip(node->data)) {//lowerBound reached
                            Logger::showMessageAndReturn("C'est le meilleur. on break le reste");
                            break; //prune remaining attributes not browsed yet
                        }
                    }
                }
                else{
                    cover->intersect(next_attributes[i]);
                    Item right_item = item(next_attributes[i], 1);
                    Array<Item> right_itemset = addItem(itemset, right_item);
                    right_node = trie->insert(right_itemset);
                    right_node = recurse(right_itemset, right_item, right_node, next_attributes, cover, depth + 1, child_ub, rlb);
                    addInfoForLowerBound(cover, right_node->data, covers, errors);
                    Error rightError = ((QueryData_Best *) right_node->data)->error;
                    right_itemset.free();
                    cover->backtrack();


                    if (query->canimprove(right_node->data, child_ub)) {

                        float remainUb = child_ub - rightError;
                        cover->intersect(next_attributes[i], false);
                        Item left_item = item(next_attributes[i], 0);
                        Array<Item> left_itemset = addItem(itemset, left_item);
                        left_node = trie->insert(left_itemset);
                        left_node = recurse(left_itemset, left_item, left_node, next_attributes, cover, depth + 1, remainUb, llb);
                        addInfoForLowerBound(cover, left_node->data, covers, errors);
                        Error leftError = ((QueryData_Best *) left_node->data)->error;
                        left_itemset.free();
                        cover->backtrack();

                        Error feature_error = leftError + rightError;
                        bool hasUpdated = query->updateData(node->data, child_ub, next_attributes[i], left_node->data, right_node->data);
                        if (hasUpdated) child_ub = feature_error;
                        if (hasUpdated) Logger::showMessageAndReturn("après cet attribut, node error = ", *nodeError, " et ub = ", child_ub);

                        if (query->canSkip(node->data)) {//lowerBound reached
                            Logger::showMessageAndReturn("C'est le meilleur. on break le reste");
                            break; //prune remaining attributes not browsed yet
                        }
                    }
                }*/


                cover->intersect(next_attributes[i], false);
                Item left_item = item(next_attributes[i], 0);
                Array<Item> left_itemset = addItem(itemset, left_item);
                TrieNode *left_node = trie->insert(left_itemset);
                Error llb = 0;
                //Error llb = computeLowerBound(cover, covers, errors);
                Error tmp1 = 0, tmp2 = 0, tmp3 = 0;
                if (covlb1){
                    Error c = errlb1, f = errlb1;
                    Supports dif1 = cover->minusMe(covlb1);
//                    difs = sumSupports(dif1);
//                    cout << "err : " << errlb1 << "per class : " << sclb1[0] << " - " << sclb1[1] << endl;
                    forEachClass(n){
                        if (sclb1[n] >= dif1[n]) c -= dif1[n];
                        else c -= sclb1[n];
                        if (sflb1[n] >= dif1[n]) f -= dif1[n];
                        else f -= sflb1[n];
                    }
                    tmp1 = min(c, f);
//                    subSupports(sclb1, dif1, sclb1);
                    deleteSupports(dif1);
//                    tmp1 = query->computeErrorValues(sclb1).error;
                    /*if (errlb1 - dif1 >= 0){
                        tmp1 = errlb1 - dif1;
                    } else if(cover->getSupport() - errlb1 - dif1 >= 0) tmp1 = cover->getSupport() - errlb1 - dif1;*/
                }
//                cout << "ltemp1 = " << tmp1 << " or old = " << (errlb1 - difs) << endl;
                if (covlb2){
                    Error c = errlb2, f = errlb2;
                    Supports dif2 = cover->minusMe(covlb2);
//                    difs = sumSupports(dif2);
//                    cout << "err : " << errlb2 << "per class : " << sclb2[0] << " - " << sclb2[1] << endl;
                    forEachClass(n){
                        if (sclb2[n] >= dif2[n]) c -= dif2[n];
                        else c -= sclb2[n];
                        if (sflb2[n] >= dif2[n]) f -= dif2[n];
                        else f -= sflb2[n];
                    }
                    tmp2 = min(c, f);
//                    subSupports(sclb2, dif2, sclb2);
                    deleteSupports(dif2);
//                    tmp2 = query->computeErrorValues(sclb2).error;
                    /*int dif2 = cover->minusMe(covlb2);
                    if (errlb2 - dif2 >= 0){
                        tmp2 = errlb2 - dif2;
                    } else if(cover->getSupport() - errlb2 - dif2 >= 0) tmp2 = cover->getSupport() - errlb2 - dif2;*/
                }
//                cout << "ltemp2 = " << tmp2 << " or old = " << (errlb2 - difs) << endl;
                if (covlb3){
                    Error c = errlb3, f = errlb3;
                    Supports dif3 = cover->minusMe(covlb3);
//                    difs = sumSupports(dif3);
//                    cout << "err : " << errlb3 << "per class : " << sclb3[0] << " - " << sclb3[1] << endl;
                    forEachClass(n){
                        if (sclb3[n] >= dif3[n]) c -= dif3[n];
                        else c -= sclb3[n];
                        if (sflb3[n] >= dif3[n]) f -= dif3[n];
                        else f -= sflb3[n];
                    }
                    tmp3 = min(c, f);
//                    subSupports(sclb3, dif3, sclb3);
                    deleteSupports(dif3);
//                    tmp3 = query->computeErrorValues(sclb3).error;
                    /*int dif3 = cover->minusMe(covlb3);
                    if (errlb3 - dif3 >= 0){
                        tmp3 = errlb3 - dif3;
                    } else if(cover->getSupport() - errlb3 - dif3 >= 0) tmp3 = cover->getSupport() - errlb3 - dif3;*/
                }
//                cout << "ltemp3 = " << tmp3 << " or old = " << (errlb3 - difs) << endl;
                llb = max(max(tmp1, tmp2), tmp3);

                left_node = recurse(left_itemset, left_item, left_node, next_attributes, cover, depth + 1, child_ub, llb);
                if (((QueryData_Best *) left_node->data)->error < FLT_MAX){
                    Error b = (((QueryData_Best *) left_node->data)->error < FLT_MAX) ? ((QueryData_Best *) left_node->data)->error : ((QueryData_Best *) left_node->data)->lowerBound;
                    if (b < FLT_MAX && b > errlb1){
                        delete [] covlb1; covlb1 = cover->getTopBitsetArray();
                        sclb1 = ((QueryData_Best *) left_node->data)->corrects;
                        sflb1 = ((QueryData_Best *) left_node->data)->falses;
                        errlb1 = b;
                        //cout << "update" << endl;
                    }
                    if (b > 0 && b < errlb2){
                        delete [] covlb2; covlb2 = cover->getTopBitsetArray();
                        sclb2 = ((QueryData_Best *) left_node->data)->corrects;
                        sflb2 = ((QueryData_Best *) left_node->data)->falses;
                        errlb2 = b;
                        //cout << "update" << endl;
                    }
                    if (cover->getSupport() > suplb){
                        delete [] covlb3; covlb3 = cover->getTopBitsetArray();
                        sclb3 = ((QueryData_Best *) left_node->data)->corrects;
                        sflb3 = ((QueryData_Best *) left_node->data)->falses;
                        suplb = cover->getSupport();
                        errlb3 = b;
                        //cout << "update" << endl;
                    }
                }

//                addInfoForLowerBound(cover, left_node->data, covers, errors);
                Error leftError = ((QueryData_Best *) left_node->data)->error;
                left_itemset.free();
                cover->backtrack();

                if (query->canimprove(left_node->data, child_ub)) {

                    float remainUb = child_ub - leftError;
                    cover->intersect(next_attributes[i]);
                    Item right_item = item(next_attributes[i], 1);
                    Array<Item> right_itemset = addItem(itemset, right_item);
                    TrieNode *right_node = trie->insert(right_itemset);
                    Error rlb = 0;
//                    Error rlb = computeLowerBound(cover, covers, errors);
                    Error tmp1 = 0, tmp2 = 0, tmp3 = 0;
                    if (covlb1){
                        Error c = errlb1, f = errlb1;
                        Supports dif1 = cover->minusMe(covlb1);
//                    difs = sumSupports(dif1);
//                    cout << "err : " << errlb1 << "per class : " << sclb1[0] << " - " << sclb1[1] << endl;
                        forEachClass(n){
                            if (sclb1[n] >= dif1[n]) c -= dif1[n];
                            else c -= sclb1[n];
                            if (sflb1[n] >= dif1[n]) f -= dif1[n];
                            else f -= sflb1[n];
                        }
                        tmp1 = min(c, f);
//                    subSupports(sclb1, dif1, sclb1);
                        deleteSupports(dif1);
//                    tmp1 = query->computeErrorValues(sclb1).error;
                        /*if (errlb1 - dif1 >= 0){
                            tmp1 = errlb1 - dif1;
                        } else if(cover->getSupport() - errlb1 - dif1 >= 0) tmp1 = cover->getSupport() - errlb1 - dif1;*/
                    }
//                cout << "ltemp1 = " << tmp1 << " or old = " << (errlb1 - difs) << endl;
                    if (covlb2){
                        Error c = errlb2, f = errlb2;
                        Supports dif2 = cover->minusMe(covlb2);
//                    difs = sumSupports(dif2);
//                    cout << "err : " << errlb2 << "per class : " << sclb2[0] << " - " << sclb2[1] << endl;
                        forEachClass(n){
                            if (sclb2[n] >= dif2[n]) c -= dif2[n];
                            else c -= sclb2[n];
                            if (sflb2[n] >= dif2[n]) f -= dif2[n];
                            else f -= sflb2[n];
                        }
                        tmp2 = min(c, f);
//                    subSupports(sclb2, dif2, sclb2);
                        deleteSupports(dif2);
//                    tmp2 = query->computeErrorValues(sclb2).error;
                        /*int dif2 = cover->minusMe(covlb2);
                        if (errlb2 - dif2 >= 0){
                            tmp2 = errlb2 - dif2;
                        } else if(cover->getSupport() - errlb2 - dif2 >= 0) tmp2 = cover->getSupport() - errlb2 - dif2;*/
                    }
//                cout << "ltemp2 = " << tmp2 << " or old = " << (errlb2 - difs) << endl;
                    if (covlb3){
                        Error c = errlb3, f = errlb3;
                        Supports dif3 = cover->minusMe(covlb3);
//                    difs = sumSupports(dif3);
//                    cout << "err : " << errlb3 << "per class : " << sclb3[0] << " - " << sclb3[1] << endl;
                        forEachClass(n){
                            if (sclb3[n] >= dif3[n]) c -= dif3[n];
                            else c -= sclb3[n];
                            if (sflb3[n] >= dif3[n]) f -= dif3[n];
                            else f -= sflb3[n];
                        }
                        tmp3 = min(c, f);
//                    subSupports(sclb3, dif3, sclb3);
                        deleteSupports(dif3);
//                    tmp3 = query->computeErrorValues(sclb3).error;
                        /*int dif3 = cover->minusMe(covlb3);
                        if (errlb3 - dif3 >= 0){
                            tmp3 = errlb3 - dif3;
                        } else if(cover->getSupport() - errlb3 - dif3 >= 0) tmp3 = cover->getSupport() - errlb3 - dif3;*/
                    }
//                cout << "ltemp3 = " << tmp3 << " or old = " << (errlb3 - difs) << endl;
                    rlb = max(max(tmp1, tmp2), tmp3);
                    right_node = recurse(right_itemset, right_item, right_node, next_attributes, cover, depth + 1, remainUb, rlb);
                    if (((QueryData_Best *) right_node->data)->error < FLT_MAX){
                        Error b = (((QueryData_Best *) right_node->data)->error < FLT_MAX) ? ((QueryData_Best *) right_node->data)->error : ((QueryData_Best *) right_node->data)->lowerBound;
                        if (b < FLT_MAX && b > errlb1){
                            delete [] covlb1; covlb1 = cover->getTopBitsetArray();
                            sclb1 = ((QueryData_Best *) right_node->data)->corrects;
                            sflb1 = ((QueryData_Best *) right_node->data)->falses;
                            errlb1 = b;
                            //cout << "update" << endl;
                        }
                        if (b > 0 && b < errlb2){
                            delete [] covlb2; covlb2 = cover->getTopBitsetArray();
                            sclb2 = ((QueryData_Best *) right_node->data)->corrects;
                            sflb2 = ((QueryData_Best *) right_node->data)->falses;
                            errlb2 = b;
                            //cout << "update" << endl;
                        }
                        if (cover->getSupport() > suplb){
                            delete [] covlb3; covlb3 = cover->getTopBitsetArray();
                            sclb3 = ((QueryData_Best *) right_node->data)->corrects;
                            sflb3 = ((QueryData_Best *) right_node->data)->falses;
                            suplb = cover->getSupport();
                            errlb3 = b;
                            //cout << "update" << endl;
                        }
                    }

//                    addInfoForLowerBound(cover, right_node->data, covers, errors);
                    Error rightError = ((QueryData_Best *) right_node->data)->error;
                    right_itemset.free();
                    cover->backtrack();

                    Error feature_error = leftError + rightError;
                    bool hasUpdated = query->updateData(node->data, child_ub, next_attributes[i], left_node->data, right_node->data);
                    if (hasUpdated) child_ub = feature_error;
                    if (hasUpdated) Logger::showMessageAndReturn("après cet attribut, node error = ", *nodeError, " et ub = ", child_ub);

                    if (query->canSkip(node->data)) {//lowerBound reached
                        Logger::showMessageAndReturn("C'est le meilleur. on break le reste");
                        break; //prune remaining attributes not browsed yet
                    }
                }
                else{
                    int dif1 = 0, dif2 = 0, dif3 = 0;
                    /*if (covlb1){
                        dif1 = cover->minusMe(covlb1);
                        if (errlb1 - dif1 >= 0){
                            dif1 = errlb1 - dif1;
                        } else if(cover->getSupport() - errlb1 - dif1 >= 0) dif1 = cover->getSupport() - errlb1 - dif1;
                    }*/
                    /*if (covlb2){
                        dif2 = cover->minusMe(covlb2);
                        if (errlb2 - dif2 >= 0){
                            dif2 = errlb2 - dif2;
                        } else if(cover->getSupport() - errlb2 - dif2 >= 0) dif2 = cover->getSupport() - errlb2 - dif2;
                    }*/
                    /*if (covlb3){
                        dif3 = cover->minusMe(covlb3);
                        if (errlb3 - dif3 >= 0){
                            dif3 = errlb3 - dif3;
                        } else if(cover->getSupport() - errlb3 - dif3 >= 0) dif3 = cover->getSupport() - errlb3 - dif3;
                    }*/
                    Error tmp = max(max(dif1, dif2), dif3);
                    if (minlb == 0) minlb = llb + tmp;
                    else if (llb + tmp < minlb) minlb = llb + tmp;
                }

                if (query->stopAfterError) {
                    if (depth == 0 && ub < FLT_MAX) {
                        if (*nodeError < ub)
                            break;
                    }
                }
            }
            delete [] covlb1;
            delete [] covlb2;
            delete [] covlb3;
            /*for (int l = 0; l < covers.size; ++l) {
                delete[] covers[l];
            }
            covers.free();
            errors.free();*/
            if (*nodeError == FLT_MAX && max(ub, minlb) > *lb) { //cache successors if solution not found
                //cout << "minlb = " << minlb << " " << *lb << endl;
                *lb = max(ub, minlb);
            }
        }


        /*if ((QueryData_Best *) node->data)->error == FLT_MAX) //cache successors if solution not found
            (QueryData_Best *) node->data)->successors = next_attributes;
        else{ //free the cache when solution found
            if ((QueryData_Best *) node->data)->successors != nullptr)
            (QueryData_Best *) node->data)->successors = nullptr;
        }*/

        Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", *nodeError);

        next_attributes.free();
//        itemset.free();
        return node;
    }


}


void LcmPruned::run() {
    query->setStartTime(clock());
    float maxError = NO_ERR;
    if (query->maxError > 0) maxError = query->maxError;
    // array of not yet visited attributes. Each attribute is represented as pair
    // the first element of the pair represent whether or not
    Array<Attribute> attributes_to_visit(nattributes, 0);
    RCover *cover = new RCover(dataReader);
    for (int attr = 0; attr < nattributes; ++attr) {
        if (cover->intersectAndSup(attr, false) >= query->minsup && cover->intersectAndSup(attr) >= query->minsup)
            attributes_to_visit.push_back(attr);
    }
    //array of items representing an itemset
    Array<Item> itemset; itemset.size = 0; itemset.elts = nullptr;
    TrieNode *node = trie->insert(itemset);
    query->realroot = recurse(itemset, NO_ITEM, node, attributes_to_visit, cover, 0, maxError);
    itemset.free(); attributes_to_visit.free(); delete cover;
    cout << "ncall: " << ncall << endl; cout << "spectime: " << spectime << endl; cout << "comptime: " << comptime << endl;
}


float LcmPruned::informationGain(Supports notTaken, Supports taken) {

    int sumSupNotTaken = sumSupports(notTaken);
    int sumSupTaken = sumSupports(taken);
    int actualDBSize = sumSupNotTaken + sumSupTaken;

    float condEntropy = 0, baseEntropy = 0;
    float priorProbNotTaken = (actualDBSize != 0) ? (float) sumSupNotTaken / actualDBSize : 0;
    float priorProbTaken = (actualDBSize != 0) ? (float) sumSupTaken / actualDBSize : 0;
    float e0 = 0, e1 = 0;

    for (int j = 0; j < dataReader->getNClasses(); ++j) {
        float p = (sumSupNotTaken != 0) ? (float) notTaken[j] / sumSupNotTaken : 0;
        float newlog = (p > 0) ? log2(p) : 0;
        e0 += -p * newlog;

        p = (float) taken[j] / sumSupTaken;
        newlog = (p > 0) ? log2(p) : 0;
        e1 += -p * newlog;

        p = (float) (notTaken[j] + taken[j]) / actualDBSize;
        newlog = (p > 0) ? log2(p) : 0;
        baseEntropy += -p * newlog;
    }
    condEntropy = priorProbNotTaken * e0 + priorProbTaken * e1;

    float actualGain = baseEntropy - condEntropy;

    return actualGain; //high error to low error when it will be put in the map. If you want to have the reverse, just return the negative value of the entropy
}


Array<Attribute> LcmPruned::getSuccessors(Array<Attribute> last_freq_attributes, RCover *cover, Item added, unordered_set<int> frequent_attr) {

    std::multimap<float, Attribute> gain;
    Array<Attribute> a_attributes2(last_freq_attributes.size, 0);
    // map<int, unordered_set<int, Hash >> control;
    // map<int, unordered_map<int, pair<int, float>, Hash>> controle;

    if (cover->getSupport() < 2 * query->minsup)
        return a_attributes2;
    int parent_sup = cover->getSupport();
    Supports parent_sup_class = copySupports(cover->getClassSupport());

    forEach (i, last_freq_attributes) {
        /*if (frequent_attr.find(last_freq_attributes[i]) != frequent_attr.end()){
            a_attributes2.push_back(last_freq_attributes[i]);
            continue;
        }*/
        if (item_attribute (added) == last_freq_attributes[i])
            continue;
        int sup_left = cover->intersectAndSup(last_freq_attributes[i], false), sup_right = parent_sup - sup_left;
        if (sup_left >= query->minsup && sup_right >= query->minsup) {
            if (query->continuous) {//continuous dataset
            } else {
                if (infoGain) {
                    Supports sup_class_left = cover->intersectAndClass(last_freq_attributes[i], false), sup_class_right = newSupports();
                    subSupports(parent_sup_class, sup_class_left, sup_class_right);
                    gain.insert(std::pair<float, Attribute>(informationGain(sup_class_left, sup_class_right), last_freq_attributes[i]));
                    deleteSupports(sup_class_left); deleteSupports(sup_class_right);
                }
                else a_attributes2.push_back(last_freq_attributes[i]);
            }
        }
    }
    deleteSupports(parent_sup_class);
    if (infoGain) {
        if (infoAsc) for (multimap<float, int>::iterator it = gain.begin(); it != gain.end(); ++it) a_attributes2.push_back(it->second); //items with low IG first
        else for (multimap<float, int>::reverse_iterator it = gain.rbegin(); it != gain.rend(); ++it) a_attributes2.push_back(it->second); //items with high IG first
    }
    if (!allDepths) infoGain = false;

    return a_attributes2;
}

Error LcmPruned::computeLowerBound(RCover* cover, Array<bitset<M>*>& covers, Array<float>& errors){
    Error tmp = 0;
    // cout << "size: " << covers.size << endl;
    for (int k = 0; k < covers.size; ++k) {
        int dif = cover->minusMee(covers[k]);
        if (errors[k] - dif > tmp)
            tmp = errors[k] - dif;
         //if (k == 10) break;
    }
    return tmp;
}

void LcmPruned::addInfoForLowerBound(RCover* cover, QueryData * node_data1, Array<bitset<M>*>& covers, Array<float>& errors){
    QueryData_Best * node_data = (QueryData_Best *) node_data1;
    Error s = node_data->lowerBound;
    if (max(node_data->error, s) < FLT_MAX && max(node_data->error, s) > 0){
        covers.push_back(cover->getTopBitsetArray());
        errors.push_back(max(node_data->error, s));
    }
}

unordered_set<int> LcmPruned::getExistingSuccessors(TrieNode *node) {
    unordered_set<int> a_attributes2;
    for (TrieEdge edge : node->edges) {
        a_attributes2.insert(item_attribute(edge.item));
    }
    return a_attributes2;
}

/*TrieNode* LcmPruned::getdepthtwotree(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb){
    //        clock_t tt = clock();
    bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;
    int root = -1, left = -1, right = -1;
    float best_root_error = ub, best_left_error1 = FLT_MAX, best_left_error2 = FLT_MAX, best_right_error1 = FLT_MAX, best_right_error2 = FLT_MAX, best_left_error = FLT_MAX, best_right_error = FLT_MAX;
    float root_leaf_error = query->computeErrorValues(cover).error, best_left_leafError = FLT_MAX, best_right_leafError = FLT_MAX;
    int best_left_class1 = -1, best_left_class2 = -1, best_right_class1 = -1, best_right_class2 = -1, best_left_class = -1, best_right_class = -1;
    forEach(i, attributes_to_visit){
        if (verbose) cout << "root test: " << attributes_to_visit[i] << endl;
        if (item_attribute(added) == attributes_to_visit[i]){
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

        float feat_left = -1, feat_right = -1;
        float best_feat_left_error = FLT_MAX, best_feat_right_error = FLT_MAX, best_feat_left_error1 = FLT_MAX, best_feat_left_error2 = FLT_MAX, best_feat_right_error1 = FLT_MAX, best_feat_right_error2 = FLT_MAX;
        float best_feat_left_leafError = FLT_MAX, best_feat_right_leafError = FLT_MAX;
        int best_feat_left_class1 = -1, best_feat_left_class2 = -1, best_feat_right_class1 = -1, best_feat_right_class2 = -1, best_feat_left_class = -1, best_feat_right_class = -1;
        int parent_sup = cover->getSupport();

        //feature to left
        cover->intersect(attributes_to_visit[i], false);
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (cover->getSupport() < query->minsup || (parent_sup - cover->getSupport()) < query->minsup){
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            cover->backtrack();
            continue;
        }
            // the feature at root cannot be splitted at left. It is then a leaf node
        else if (cover->getSupport() < 2 * query->minsup){
            ErrorValues ev = query->computeErrorValues(cover);
            best_feat_left_error = ev.error;
            best_feat_left_class = ev.maxclass;
            if (verbose) cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << best_feat_left_error << " on backtrack" << endl;
            cover->backtrack();
        }
            // the root node can theorically be split at left
        else {
            if (verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            ErrorValues ev = query->computeErrorValues(cover);
            best_feat_left_error = min(ev.error, best_root_error);
            best_feat_left_leafError = ev.error;
            best_feat_left_class = ev.maxclass;
            if (ev.error != lb){
                float tmp = best_feat_left_error;
                forEach(j, attributes_to_visit) {
                    if (verbose) cout << "left test: " << attributes_to_visit[j] << endl;
                    if (item_attribute(added) == attributes_to_visit[j]) {
                        if (verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }
                    parent_sup = cover->getSupport();
                    Supports parent_sup_class = cover->getSupportPerClass().first;
                    cover->intersect(attributes_to_visit[j], false);
                    // the root node can in practice be split into two children
                    if (cover->getSupport() >= query->minsup && (parent_sup - cover->getSupport()) >= query->minsup) {
                        if (verbose) cout << "le left testé peut splitter. on le regarde" << endl;
                        ev = query->computeErrorValues(cover);
                        float tmp_left_error1 = ev.error;
                        int tmp_left_class1 = ev.maxclass;
                        if (verbose) cout << "le left a gauche produit une erreur de " << tmp_left_error1 << endl;
//                        cover->backtrack();


                        if (tmp_left_error1 >= min(best_root_error, best_feat_left_error)) {
                            if (verbose) cout << "l'erreur gauche du left montre rien de bon. best root: " << best_root_error << " best left: " << best_feat_left_error << " Un autre left..." << endl;
                            cover->backtrack();
                            continue;
                        }

                        Supports droite_sup_class = cover->getSupportPerClass().first;
                        for (int k = 0; k < nclasses; ++k) {
                            if (verbose) cout << "parent[" << k <<"] = " << parent_sup_class[k] << " gauche[" << k <<"] = " << droite_sup_class[k] << endl;
                            droite_sup_class[k] = parent_sup_class[k] - droite_sup_class[k];
                        }
                        cover->backtrack();

//                        cover->intersect(attributes_to_visit[j]);
//                        ev = query->computeErrorValues(cover);
                        ev = query->computeErrorValues(droite_sup_class);
                        deleteSupports(parent_sup_class);
                        float tmp_left_error2 = ev.error;
                        int tmp_left_class2 = ev.maxclass;
                        if (verbose) cout << "le left a droite produit une erreur de " << tmp_left_error2 << endl;
//                        cover->backtrack();
                        if (tmp_left_error1 + tmp_left_error2 < min(best_root_error, best_feat_left_error)) {
                            best_feat_left_error = tmp_left_error1 + tmp_left_error2;
                            if (verbose) cout << "ce left ci donne une meilleure erreur que les précédents left: " << best_feat_left_error << endl;
                            best_feat_left_error1 = tmp_left_error1;
                            best_feat_left_error2 = tmp_left_error2;
                            best_feat_left_class1 = tmp_left_class1;
                            best_feat_left_class2 = tmp_left_class2;
                            feat_left = attributes_to_visit[j];
                            if (best_feat_left_error == lb) break;
                        }
                        else{
                            if (verbose) cout << "l'erreur du left = " << tmp_left_error1 + tmp_left_error2 << " n'ameliore pas l'existant. Un autre left..." << endl;
                        }
                    }
                    else {
                        if (verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                        cover->backtrack();
                    }
                }
                if (best_feat_left_error == tmp){
                    if (verbose) cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
                }
            }
            else {
                if (verbose) cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
            }

            cover->backtrack();
        }


        //feature to right
        if (best_feat_left_error < best_root_error){
            if (verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            cover->intersect(attributes_to_visit[i]);
            // the feature at root cannot be split at right. It is then a leaf node
            if (cover->getSupport() < 2 * query->minsup){
                ErrorValues ev = query->computeErrorValues(cover);
                best_feat_right_error = ev.error;
                best_feat_right_class = ev.maxclass;
                if (verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << best_feat_right_error << " on backtrack" << endl;
                cover->backtrack();
            }
            else {
                if (verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                ErrorValues ev = query->computeErrorValues(cover);
                best_feat_right_error = min(ev.error, (best_root_error - best_feat_left_error));
                best_feat_right_leafError = ev.error;
                best_feat_right_class = ev.maxclass;
                float tmp = best_feat_right_error;
                if (ev.error != lb){
                    forEach(j, attributes_to_visit) {
                        if (verbose) cout << "right test: " << attributes_to_visit[j] << endl;
                        if (item_attribute(added) == attributes_to_visit[j]) {
//                                cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }
                        parent_sup = cover->getSupport();
                        Supports parent_sup_class = cover->getSupportPerClass().first;
                        cover->intersect(attributes_to_visit[j], false);
                        // the root node can in practice be split into two children
                        if (cover->getSupport() >= query->minsup && (parent_sup - cover->getSupport()) >= query->minsup) {
                            if (verbose) cout << "le right testé peut splitter. on le regarde" << endl;
                            ev = query->computeErrorValues(cover);
                            float tmp_right_error1 = ev.error;
                            int tmp_right_class1 = ev.maxclass;
                            if (verbose) cout << "le right a gauche produit une erreur de " << tmp_right_error1 << endl;
//                            cover->backtrack();

                            if (tmp_right_error1 >= min((best_root_error - best_feat_left_error), best_feat_right_error)){
                                if (verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                cover->backtrack();
                                continue;
                            }

                            Supports droite_sup_class = cover->getSupportPerClass().first;
                            for (int k = 0; k < nclasses; ++k) {
                                if (verbose) cout << "parent[" << k <<"] = " << parent_sup_class[k] << " gauche[" << k <<"] = " << droite_sup_class[k] << endl;
                                droite_sup_class[k] = parent_sup_class[k] - droite_sup_class[k];
                            }
                            cover->backtrack();

//                            cover->intersect(attributes_to_visit[j]);
//                            ev = query->computeErrorValues(cover);
                            ev = query->computeErrorValues(droite_sup_class);
                            deleteSupports(parent_sup_class);
                            float tmp_right_error2 = ev.error;
                            int tmp_right_class2 = ev.maxclass;
                            if (verbose) cout << "le right a droite produit une erreur de " << tmp_right_error2 << endl;
//                            cover->backtrack();
                            if (tmp_right_error1 + tmp_right_error2 < min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                best_feat_right_error = tmp_right_error1 + tmp_right_error2;
                                if (verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << best_feat_right_error << endl;
                                best_feat_right_error1 = tmp_right_error1;
                                best_feat_right_error2 = tmp_right_error2;
                                best_feat_right_class1 = tmp_right_class1;
                                best_feat_right_class2 = tmp_right_class2;
                                feat_right = attributes_to_visit[j];
                                if (best_feat_right_error == lb) break;
                            }
                            else{
                                if (verbose) cout << "l'erreur du right = " << tmp_right_error1 + tmp_right_error2 << " n'ameliore pas l'existant. Un autre right..." << endl;
                            }
                        }
                        else {
                            if (verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                            cover->backtrack();
                        }
                    }
                    if (best_feat_right_error == tmp){
                        if (verbose) cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: " << best_feat_right_error << endl;
                    }
                }
                else {
                    if (verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << best_feat_left_error << endl;
                }
                cover->backtrack();
            }

            if (best_feat_left_error + best_feat_right_error < best_root_error){
                best_root_error = best_feat_left_error + best_feat_right_error;
                if (verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << best_root_error << endl;
                best_left_error = best_feat_left_error;
                best_right_error = best_feat_right_error;
                best_left_leafError = best_feat_left_leafError;
                best_right_leafError = best_feat_right_leafError;
                best_left_class = best_feat_left_class;
                best_right_class = best_feat_right_class;
                left = feat_left;
                right = feat_right;
                root = attributes_to_visit[i];
                best_left_error1 = best_feat_left_error1;
                best_left_error2 = best_feat_left_error2;
                best_right_error1 = best_feat_right_error1;
                best_right_error2 = best_feat_right_error2;
                best_left_class1 = best_feat_left_class1;
                best_left_class2 = best_feat_left_class2;
                best_right_class1 = best_feat_right_class1;
                best_right_class2 = best_feat_right_class2;
            }
        }
    }
    if (verbose) cout << "root: " << root << " left: " << left << " right: " << right << endl;
    if (verbose) cout << "le1: " << best_left_error1 << " le2: " << best_left_error2 << " re1: " << best_right_error1 << " re2: " << best_right_error2 << endl;
    if (verbose) cout << "ble: " << best_left_error << " bre: " << best_right_error << " broe: " << best_root_error << endl;
    if (verbose) cout << "lc1: " << best_left_class1 << " lc2: " << best_left_class2 << " rc1: " << best_right_class1 << " rc2: " << best_right_class2 << endl;
    if (verbose) cout << "blc: " << best_left_class << " brc: " << best_right_class << endl;
//    if (verbose) cout << "temps: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;

    if (root != -1){
//            cout << "cc0" << endl;
        //insert root to left
        Array<Item > root_neg;
        root_neg.alloc(itemset.size + 1);
        addItem(itemset, item(root, 0), root_neg);
        TrieNode* root_neg_node = trie->insert(root_neg);
        root_neg_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_neg_node->data)->error = best_left_error;
        if (left == -1) {
            ((QueryData_Best *) root_neg_node->data)->test = best_left_class;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_error;
            ((QueryData_Best *) root_neg_node->data)->size = 1;
            ((QueryData_Best *) root_neg_node->data)->left = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_neg_node->data)->test = left;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_leafError;
            ((QueryData_Best *) root_neg_node->data)->size = 3;
        }
//            cout << "cc1*" << endl;

        //insert root to right
        Array<Item > root_pos;
        root_pos.alloc(itemset.size + 1);
        addItem(itemset, item(root, 1), root_pos);
        TrieNode* root_pos_node = trie->insert(root_pos);
        root_pos_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_pos_node->data)->error = best_right_error;
        if (right == -1) {
            ((QueryData_Best *) root_pos_node->data)->test = best_right_class;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_error;
            ((QueryData_Best *) root_pos_node->data)->size = 1;
            ((QueryData_Best *) root_pos_node->data)->left = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_pos_node->data)->test = right;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_leafError;
            ((QueryData_Best *) root_pos_node->data)->size = 3;
        }

//        itemset.free();
//            cout << "cc0*" << endl;

        if (left != -1){
//                cout << "cc00" << endl;
            //insert left neg
            Array<Item > left_neg;
            left_neg.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 0), left_neg);
            TrieNode* left_neg_node = trie->insert(left_neg);
            left_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_neg_node->data)->error = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->leafError = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->test = best_left_class1;
            ((QueryData_Best *) left_neg_node->data)->size = 1;
            ((QueryData_Best *) left_neg_node->data)->left = nullptr;
            ((QueryData_Best *) left_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->left = (QueryData_Best *) left_neg_node->data;

            //insert left pos
            Array<Item > left_pos;
            left_pos.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 1), left_pos);
            TrieNode* left_pos_node = trie->insert(left_pos);
            left_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_pos_node->data)->error = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->leafError = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->test = best_left_class2;
            ((QueryData_Best *) left_pos_node->data)->size = 1;
            ((QueryData_Best *) left_pos_node->data)->left = nullptr;
            ((QueryData_Best *) left_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = (QueryData_Best *) left_pos_node->data;

            left_neg.free();
            left_pos.free();
            root_neg.free();
        }

        if (right != -1){
//                cout << "cc000" << endl;
            //insert right neg
            Array<Item > right_neg;
            right_neg.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 0), right_neg);
            TrieNode* right_neg_node = trie->insert(right_neg);
            right_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_neg_node->data)->error = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->leafError = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->test = best_right_class1;
            ((QueryData_Best *) right_neg_node->data)->size = 1;
            ((QueryData_Best *) right_neg_node->data)->left = nullptr;
            ((QueryData_Best *) right_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->left = (QueryData_Best *) right_neg_node->data;

            //insert right pos
            Array<Item > right_pos;
            right_pos.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 1), right_pos);
            TrieNode* right_pos_node = trie->insert(right_pos);
            right_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_pos_node->data)->error = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->leafError = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->test = best_right_class2;
            ((QueryData_Best *) right_pos_node->data)->size = 1;
            ((QueryData_Best *) right_pos_node->data)->left = nullptr;
            ((QueryData_Best *) right_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = (QueryData_Best *) right_pos_node->data;

            right_neg.free();
            right_pos.free();
            root_pos.free();
        }
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = best_root_error;
        ((QueryData_Best *) node->data)->leafError = root_leaf_error;
        ((QueryData_Best *) node->data)->test = root;
        ((QueryData_Best *) node->data)->size = ((QueryData_Best *) root_neg_node->data)->size + ((QueryData_Best *) root_pos_node->data)->size + 1;
        ((QueryData_Best *) node->data)->left = (QueryData_Best *) root_neg_node->data;
        ((QueryData_Best *) node->data)->right = (QueryData_Best *) root_pos_node->data;
//            cout << "cc1" << endl;
        return node;
    }
    else{
        //error not lower than ub
//            cout << "cale" << endl;
        ErrorValues ev = query->computeErrorValues(cover);
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = FLT_MAX;
        ((QueryData_Best *) node->data)->leafError = ev.error;
        ((QueryData_Best *) node->data)->test = ev.maxclass;
        ((QueryData_Best *) node->data)->size = 1;
        ((QueryData_Best *) node->data)->left = nullptr;
        ((QueryData_Best *) node->data)->right = nullptr;
//            cout << "cc2" << endl;
        return node;
    }
}*/



/*struct TwoTreeInfo{
    QueryData_Best root_node, left_node, right_node;
    Attribute root_attr = -1, l_attr = -1, r_attr = -1;
    Error root_lerror = -1, l_lerror = -1, r_lerror = -1;
    Error root_error = FLT_MAX, l_error = FLT_MAX, r_error = FLT_MAX, l1_error = FLT_MAX, l2_error = FLT_MAX, r1_error = FLT_MAX, r2_error = FLT_MAX;
    Class root_class = -1, l_class = -1, r_class = -1, l1_class = -1, l2_class = -1, r1_class = -1, r2_class = -1;
};*/

struct TwoTreeInfo{
    QueryData_Best root_node, left_node, right_node;
    //Attribute root_attr = -1, l_attr = -1, r_attr = -1;
    //Error root_lerror = -1, l_lerror = -1, r_lerror = -1;
    //Error root_error = FLT_MAX, l_error = FLT_MAX, r_error = FLT_MAX, l1_error = FLT_MAX, l2_error = FLT_MAX, r1_error = FLT_MAX, r2_error = FLT_MAX;
    //Class root_class = -1, l_class = -1, r_class = -1, l1_class = -1, l2_class = -1, r1_class = -1, r2_class = -1;
};

void fillnodedata(QueryData_Best* node, ErrorValues ev){
    //node.test = ev.maxclass;
    node->leafError = ev.error;
    node->size = 1;
}


void copynodedata(QueryData_Best* node, QueryData_Best other){
    //node.test = ev.maxclass;
    node->leafError = other.leafError;
    node->size = 1;
}


/*TrieNode* LcmPruned::getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb) {
    ncall += 1;
    if (ub <= lb){
        // cout << "cc" << endl;
        node->data = query->initData(cover);
        ((QueryData_Best *) node->data)->error = ((QueryData_Best *) node->data)->leafError;
        return node;
    }
    clock_t tt = clock();
    bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;

    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    int root_sup = cover->getSupport();

    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for (int m = 0; m < attributes_to_visit.size; ++m) {
        if (item_attribute(added) == attributes_to_visit[m]) continue;
        attr.push_back(attributes_to_visit[m]);
    }
    clock_t ttt = clock();
    Supports **sups = new Supports *[attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        sups[l] = new Supports[attr.size()];
        cover->intersect(attr[l]);
        sups[l][l] = cover->getSupportPerClass();
        for (int i = l + 1; i < attr.size(); ++i) sups[l][i] = cover->intersectAndClass(attr[i]);
        cover->backtrack();
    }
    comptime += (clock() - ttt) / (float) CLOCKS_PER_SEC;
    //cout << "temps comp: " << (clock() - ttt) / (float) CLOCKS_PER_SEC << " " << endl;
    //exit(0);

    //QueryData_Best* besttree = new QueryData_Best();
    //Error best_error = min(ub, ((QueryData_Best*)node->data)->leafError);
    node->data = query->initData(cover);
    ((QueryData_Best *) node->data)->error = ((QueryData_Best *) node->data)->leafError;

    //cout << "test" << endl;

    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;
        if (item_attribute(added) == attr[i]) {
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

        Supports idsc = sups[i][i];
        int ids = sumSupports(idsc);
        Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc);
        int igs = root_sup - ids;

        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup) {
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }

        //cout << "check i: " << i << endl;

        ErrorValues ev;
        QueryData_Best *feattree = new QueryData_Best();
        copynodedata(feattree, *(QueryData_Best *) node->data);
        feattree->test = attr[i];
        feattree->left = new QueryData_Best();
        ev = query->computeErrorValues(igsc);
        fillnodedata(feattree->left, ev);
        feattree->left->test = ev.maxclass;
        feattree->left->error = ev.error;
        feattree->left->size = 1;
        feattree->right = new QueryData_Best();
        ev = query->computeErrorValues(idsc);
        fillnodedata(feattree->right, ev);
        feattree->right->test = ev.maxclass;
        feattree->right->error = ev.error;
        feattree->right->size = 1;
        feattree->size = 3;
        feattree->error = feattree->left->error + feattree->right->error;

        for (int j = 0; j < attr.size(); ++j) {
            Supports jdsc = sups[j][j], idjdsc = sups[min(i, j)][max(i, j)], igjdsc = newSupports();
            subSupports(jdsc, idjdsc, igjdsc);
            int jds = sumSupports(jdsc), idjds = sumSupports(idjdsc), igjds = sumSupports(igjdsc);
            int igjgs = igs - igjds;
            Supports igjgsc = newSupports();
            subSupports(igsc, igjdsc, igjgsc);
            Supports idjgsc = newSupports();
            subSupports(idsc, idjdsc, idjgsc);
            int idjgs = ids - idjds;

            if (igjgs >= query->minsup && igjds >= query->minsup) {
                ErrorValues l1 = query->computeErrorValues(igjgsc), l2 = query->computeErrorValues(igjdsc);
                if (l1.error + l2.error < feattree->left->error) {
                    feattree->size = feattree->size - feattree->left->size + 3;
                    feattree->error = feattree->error - feattree->left->error + l1.error + l2.error;
                    feattree->left->test = attr[j];
                    feattree->left->error = l1.error + l2.error;
                    feattree->left->size = 3;
                    QueryData_Best *leaf1 = new QueryData_Best();
                    leaf1->test = l1.maxclass;
                    leaf1->error = l1.error;
                    leaf1->leafError = l1.error;
                    QueryData_Best *leaf2 = new QueryData_Best();
                    leaf2->test = l2.maxclass;
                    leaf2->error = l2.error;
                    leaf2->leafError = l2.error;
                    if (feattree->left->left != nullptr) {
                        delete feattree->left->left;
                        delete feattree->left->right;
                    }
                    feattree->left->left = leaf1;
                    feattree->left->right = leaf2;
                }
            }

            if (idjgs >= query->minsup && idjds >= query->minsup) {
                ErrorValues r1 = query->computeErrorValues(idjgsc), r2 = query->computeErrorValues(idjdsc);
                if (r1.error + r2.error < feattree->right->error) {
                    feattree->size = feattree->size - feattree->right->size + 3;
                    feattree->error = feattree->error - feattree->right->error + r1.error + r2.error;
                    feattree->right->test = attr[j];
                    feattree->right->error = r1.error + r2.error;
                    feattree->right->size = 3;
                    QueryData_Best *leaf1 = new QueryData_Best();
                    leaf1->test = r1.maxclass;
                    leaf1->error = r1.error;
                    leaf1->leafError = r1.error;
                    QueryData_Best *leaf2 = new QueryData_Best();
                    leaf2->test = r2.maxclass;
                    leaf2->error = r2.error;
                    leaf2->leafError = r2.error;
                    if (feattree->right->left != nullptr) {
                        delete feattree->right->left;
                        delete feattree->right->right;
                    }
                    feattree->right->left = leaf1;
                    feattree->right->right = leaf2;
                }
            }
        }


        if (feattree->error < ((QueryData_Best *) node->data)->error) {
            //remove current tree
            if (((QueryData_Best *) node->data)->left != nullptr) {
                if (((QueryData_Best *) node->data)->left->left != nullptr) {
                    delete ((QueryData_Best *) node->data)->left->left;
                    delete ((QueryData_Best *) node->data)->left->right;
                }
                delete ((QueryData_Best *) node->data)->left;

                if (((QueryData_Best *) node->data)->right->left != nullptr) {
                    delete ((QueryData_Best *) node->data)->right->left;
                    delete ((QueryData_Best *) node->data)->right->right;
                }
                delete ((QueryData_Best *) node->data)->right;
            }
            delete node->data;
            node->data = (QueryData *) feattree;
        }
    }
    if (((QueryData_Best *) node->data)->left != nullptr) {
        Array<Item> itemset_left, itemset_right;
        itemset_left.alloc(itemset.size + 1);
        itemset_right.alloc(itemset.size + 1);
        TrieNode *node_;

        addItem(itemset, item(((QueryData_Best *) node->data)->test, 0), itemset_left);
        node_ = trie->insert(itemset_left);
        node_->data = (QueryData *) ((QueryData_Best *) node->data)->left;

        addItem(itemset, item(((QueryData_Best *) node->data)->test, 1), itemset_right);
        node_ = trie->insert(itemset_right);
        node_->data = (QueryData *) ((QueryData_Best *) node->data)->right;

        if (((QueryData_Best *) node->data)->left->left != nullptr) {
            Array<Item> itemset_;
            itemset_.alloc(itemset_left.size + 1);

            addItem(itemset_left, item(((QueryData_Best *) node->data)->left->test, 0), itemset_);
            node_ = trie->insert(itemset_);
            node_->data = (QueryData *) ((QueryData_Best *) node->data)->left->left;

            addItem(itemset_left, item(((QueryData_Best *) node->data)->left->test, 1), itemset_);
            node_ = trie->insert(itemset_);
            node_->data = (QueryData *) ((QueryData_Best *) node->data)->left->right;

            itemset_.free();
        }

        if (((QueryData_Best *) node->data)->right->left != nullptr) {
            Array<Item> itemset_;
            itemset_.alloc(itemset_right.size + 1);

            addItem(itemset_right, item(((QueryData_Best *) node->data)->right->test, 0), itemset_);
            node_ = trie->insert(itemset_);
            node_->data = (QueryData *) ((QueryData_Best *) node->data)->right->left;

            addItem(itemset_right, item(((QueryData_Best *) node->data)->right->test, 1), itemset_);
            node_ = trie->insert(itemset_);
            node_->data = (QueryData *) ((QueryData_Best *) node->data)->right->right;

            itemset_.free();
        }

        itemset_left.free();
        itemset_right.free();
    }
    spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
    return node;
}*/






/*TrieNode* LcmPruned::getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb){
    ncall += 1; //cout << "ncall: " << ncall;
    clock_t tt = clock(); //cout << "tempss: " << (clock() - tt) / (float) CLOCKS_PER_SEC;
    bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;

    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    int root_sup = cover->getSupport();

    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for (int m = 0; m < attributes_to_visit.size; ++m) {
        if (item_attribute(added) == attributes_to_visit[m]) continue;
        attr.push_back(attributes_to_visit[m]);
    }
    clock_t ttt = clock();
    Supports** sups = new Supports*[attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        sups[l] = new Supports[attr.size()];
        cover->intersect(attr[l]);
        sups[l][l] = cover->getSupportPerClass();
        if (cover->getSupport() < 2 * query->minsup && (root_sup - cover->getSupport()) < 2 * query->minsup){
            cover->backtrack();
            continue;
        }
//        parallel_for(attr.size() - l - 1, [&](int start, int end){
//            for(int i = start + (l+1); i < end + (l+1); ++i)
//                sups[l][i] = cover->intersectAndClass(attr[i]);
//        }, false );
        for (int i = l+1; i < attr.size(); ++i) {
            sups[l][i] = cover->intersectAndClass(attr[i]);
        }
        cover->backtrack();
    }
    comptime += (clock() - ttt) / (float) CLOCKS_PER_SEC;


    //int root = -1, left = -1, right = -1;
    TwoTreeInfo treeinfo;
    //treeinfo.root_error = ub;
    //float best_root_error = ub, best_left_error1 = FLT_MAX, best_left_error2 = FLT_MAX, best_right_error1 = FLT_MAX, best_right_error2 = FLT_MAX, best_left_error = FLT_MAX, best_right_error = FLT_MAX;
    //float root_leaf_error = query->computeErrorValues(cover).error, best_left_leafError = FLT_MAX, best_right_leafError = FLT_MAX;
    //int best_left_class1 = -1, best_left_class2 = -1, best_right_class1 = -1, best_right_class2 = -1, best_left_class = -1, best_right_class = -1;
    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;
        if (item_attribute(added) == attr[i]){
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

//        float feat_left = -1, feat_right = -1;
//        float best_feat_left_error = FLT_MAX, best_feat_right_error = FLT_MAX, best_feat_left_error1 = FLT_MAX, best_feat_left_error2 = FLT_MAX, best_feat_right_error1 = FLT_MAX, best_feat_right_error2 = FLT_MAX;
//        float best_feat_left_leafError = FLT_MAX, best_feat_right_leafError = FLT_MAX;
//        int best_feat_left_class1 = -1, best_feat_left_class2 = -1, best_feat_right_class1 = -1, best_feat_right_class2 = -1, best_feat_left_class = -1, best_feat_right_class = -1;

        Supports idsc = sups[i][i];
        int ids = sumSupports(idsc);
        Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc);
        int igs = root_sup - ids;

        TwoTreeInfo tmpTreeInfo;
        tmpTreeInfo.root_error = ub;
        bool search_left = true, search_right = true;
        ErrorValues ev_left, ev_right;
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup){
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }
        // the feature at root cannot be splitted at left. It is then a leaf node
        if (igs < 2 * query->minsup || ids < 2 * query->minsup){
            if(igs < 2 * query->minsup){
                search_left = false;
                ev_left = query->computeErrorValues(igsc);
                if (ev_left.error < tmpTreeInfo.l_error || (ev_left.error == tmpTreeInfo.l_error && tmpTreeInfo.l_class != -1)) {
                    tmpTreeInfo.l_attr = -1;
                    tmpTreeInfo.l_error = ev_left.error;
                    tmpTreeInfo.l_class = ev_left.maxclass;
                }
                if (verbose) cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << treeinfo.l_error << endl;
            }
            else{
                search_right = false;
                ev_right = query->computeErrorValues(igsc);
                if (ev_right.error < tmpTreeInfo.r_error || (ev_right.error == tmpTreeInfo.r_error && tmpTreeInfo.r_class != -1)) {
                    tmpTreeInfo.l_attr = -1;
                    tmpTreeInfo.r_error = ev_right.error;
                    tmpTreeInfo.r_class = ev_right.maxclass;
                }
                if (verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << treeinfo.r_error << endl;
            }
        }
        if (search_left) {
            ev_left = query->computeErrorValues(igsc);
            tmpTreeInfo.l_error = min(ev_left.error, tmpTreeInfo.root_error);
            tmpTreeInfo.l_class = ev_left.maxclass;
            if (tmpTreeInfo.l_error == lb) {
                search_left = false;
                tmpTreeInfo.l_error = min(ev_left.error, tmpTreeInfo.root_error);
                tmpTreeInfo.l_class = ev_left.maxclass;
            }
        }
        if (search_right) {
            ev_right = query->computeErrorValues(idsc);
            tmpTreeInfo.r_error = min(ev_right.error, tmpTreeInfo.root_error);
            tmpTreeInfo.r_class = ev_right.maxclass;
            if (tmpTreeInfo.r_error == lb) search_right = false;
        }
        if (verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
        // at worst it can't in practice and error will be considered as leaf node
        // so the error is initialized at this case
//        ErrorValues ev = query->computeErrorValues(igsc);
//        best_feat_left_error = min(ev.error, best_root_error);
//        best_feat_left_leafError = ev.error;
//        best_feat_left_class = ev.maxclass;
//        if (ev.error != lb){
            float tmp = best_feat_left_error;
            for (int j = 0; j < attr.size(); ++j) {
                if (verbose) cout << "left test: " << attr[j] << endl;
                if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
                    if (verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                    continue;
                }
                Supports jdsc = sups[j][j], idjdsc = sups[min(i,j)][max(i,j)], igjdsc = newSupports(); subSupports(jdsc, idjdsc, igjdsc);
                int jds = sumSupports(jdsc), idjds = sumSupports(idjdsc), igjds = sumSupports(igjdsc); int igjgs = igs - igjds;

                // the root node can in practice be split into two children
                if (igjgs >= query->minsup && igjds >= query->minsup) {
                    if (verbose) cout << "le left testé peut splitter. on le regarde" << endl;

                    ev = query->computeErrorValues(igjdsc);
                    float tmp_left_error2 = ev.error;
                    int tmp_left_class2 = ev.maxclass;
                    if (verbose) cout << "le left a droite produit une erreur de " << tmp_left_error2 << endl;

                    if (tmp_left_error2 >= min(best_root_error, best_feat_left_error)) {
                        if (verbose) cout << "l'erreur gauche du left montre rien de bon. best root: " << best_root_error << " best left: " << best_feat_left_error << " Un autre left..." << endl;
                        continue;
                    }

                    Supports igjgsc = newSupports(); subSupports(igsc, igjdsc, igjgsc);
//                        cout << "igsc = " << igsc[0] << ", " << igsc[1] << endl;
//                        cout << "igjdsc = " << igjdsc[0] << ", " << igjdsc[1] << endl;
//                        cout << "igjgsc = " << igjgsc[0] << ", " << igjgsc[1] << endl;
                    ev = query->computeErrorValues(igjgsc);
                    float tmp_left_error1 = ev.error;
                    int tmp_left_class1 = ev.maxclass;
                    if (verbose) cout << "le left a gauche produit une erreur de " << tmp_left_error1 << endl;

                    if (tmp_left_error1 + tmp_left_error2 < min(best_root_error, best_feat_left_error)) {
                        best_feat_left_error = tmp_left_error1 + tmp_left_error2;
                        if (verbose) cout << "ce left ci donne une meilleure erreur que les précédents left: " << best_feat_left_error << endl;
                        best_feat_left_error1 = tmp_left_error1;
                        best_feat_left_error2 = tmp_left_error2;
                        best_feat_left_class1 = tmp_left_class1;
                        best_feat_left_class2 = tmp_left_class2;
                        feat_left = attr[j];
                        if (best_feat_left_error == lb) break;
                    }
                    else if (verbose) cout << "l'erreur du left = " << tmp_left_error1 + tmp_left_error2 << " n'ameliore pas l'existant. Un autre left..." << endl;
                    deleteSupports(igjgsc);
                }
                else if (verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                deleteSupports(igjdsc);
            }
            if (best_feat_left_error == tmp && verbose) cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
//        }
//        else {
//            if (verbose) cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
//        }


        //feature to right
        if (best_feat_left_error < best_root_error){
            if (verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * query->minsup){
                ErrorValues ev = query->computeErrorValues(idsc);
                best_feat_right_error = ev.error;
                best_feat_right_class = ev.maxclass;
                if (verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << best_feat_right_error << " on backtrack" << endl;
            }
            else {
                if (verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                ErrorValues ev = query->computeErrorValues(idsc);
                best_feat_right_error = min(ev.error, (best_root_error - best_feat_left_error));
                best_feat_right_leafError = ev.error;
                best_feat_right_class = ev.maxclass;
                float tmp = best_feat_right_error;
                if (ev.error != lb){
                    for (int j = 0; j < attr.size(); ++j) {
                        if (verbose) cout << "right test: " << attr[j] << endl;
                        if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
//                                cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }

                        Supports idjdsc = sups[min(i,j)][max(i,j)], idjgsc = newSupports(); subSupports(idsc, idjdsc, idjgsc);
                        int idjds = sumSupports(idjdsc), idjgs = sumSupports(idjgsc);

                        // the root node can in practice be split into two children
                        if (idjgs >= query->minsup && idjds >= query->minsup) {
                            if (verbose) cout << "le right testé peut splitter. on le regarde" << endl;
                            ev = query->computeErrorValues(idjgsc);
                            float tmp_right_error1 = ev.error;
                            int tmp_right_class1 = ev.maxclass;
                            if (verbose) cout << "le right a gauche produit une erreur de " << tmp_right_error1 << endl;

                            if (tmp_right_error1 >= min((best_root_error - best_feat_left_error), best_feat_right_error)){
                                if (verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                continue;
                            }

                            ev = query->computeErrorValues(idjdsc);
                            float tmp_right_error2 = ev.error;
                            int tmp_right_class2 = ev.maxclass;
                            if (verbose) cout << "le right a droite produit une erreur de " << tmp_right_error2 << endl;
                            if (tmp_right_error1 + tmp_right_error2 < min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                best_feat_right_error = tmp_right_error1 + tmp_right_error2;
                                if (verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << best_feat_right_error << endl;
                                best_feat_right_error1 = tmp_right_error1;
                                best_feat_right_error2 = tmp_right_error2;
                                best_feat_right_class1 = tmp_right_class1;
                                best_feat_right_class2 = tmp_right_class2;
                                feat_right = attr[j];
                                if (best_feat_right_error == lb) break;
                            }
                            else if (verbose) cout << "l'erreur du right = " << tmp_right_error1 + tmp_right_error2 << " n'ameliore pas l'existant. Un autre right..." << endl;
                        }
                        else if (verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        deleteSupports(idjgsc);
                    }
                    if (best_feat_right_error == tmp) if (verbose) cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: " << best_feat_right_error << endl;
                }
                else if (verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << best_feat_left_error << endl;
            }

            if (best_feat_left_error + best_feat_right_error < best_root_error){
                best_root_error = best_feat_left_error + best_feat_right_error;
                if (verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << best_root_error << endl;
                best_left_error = best_feat_left_error;
                best_right_error = best_feat_right_error;
                best_left_leafError = best_feat_left_leafError;
                best_right_leafError = best_feat_right_leafError;
                best_left_class = best_feat_left_class;
                best_right_class = best_feat_right_class;
                left = feat_left;
                right = feat_right;
                root = attr[i];
                best_left_error1 = best_feat_left_error1;
                best_left_error2 = best_feat_left_error2;
                best_right_error1 = best_feat_right_error1;
                best_right_error2 = best_feat_right_error2;
                best_left_class1 = best_feat_left_class1;
                best_left_class2 = best_feat_left_class2;
                best_right_class1 = best_feat_right_class1;
                best_right_class2 = best_feat_right_class2;
            }
        }
        deleteSupports(igsc);
    }
    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) {
            deleteSupports(sups[k][i]);
        }
    }
    if (verbose) cout << "root: " << root << " left: " << left << " right: " << right << endl;
    if (verbose) cout << "le1: " << best_left_error1 << " le2: " << best_left_error2 << " re1: " << best_right_error1 << " re2: " << best_right_error2 << endl;
    if (verbose) cout << "ble: " << best_left_error << " bre: " << best_right_error << " broe: " << best_root_error << endl;
    if (verbose) cout << "lc1: " << best_left_class1 << " lc2: " << best_left_class2 << " rc1: " << best_right_class1 << " rc2: " << best_right_class2 << endl;
    if (verbose) cout << "blc: " << best_left_class << " brc: " << best_right_class << endl;
//    cout << "temps find: " << (clock() - tt) / (float) CLOCKS_PER_SEC << " ";

    if (root != -1){
//            cout << "cc0" << endl;
        //insert root to left
        Array<Item > root_neg;
        root_neg.alloc(itemset.size + 1);
        addItem(itemset, item(root, 0), root_neg);
        TrieNode* root_neg_node = trie->insert(root_neg);
        root_neg_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_neg_node->data)->error = best_left_error;
        if (left == -1) {
            ((QueryData_Best *) root_neg_node->data)->test = best_left_class;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_error;
            ((QueryData_Best *) root_neg_node->data)->size = 1;
            ((QueryData_Best *) root_neg_node->data)->left = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_neg_node->data)->test = left;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_leafError;
            ((QueryData_Best *) root_neg_node->data)->size = 3;
        }
//            cout << "cc1*" << endl;

        //insert root to right
        Array<Item > root_pos;
        root_pos.alloc(itemset.size + 1);
        addItem(itemset, item(root, 1), root_pos);
        TrieNode* root_pos_node = trie->insert(root_pos);
        root_pos_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_pos_node->data)->error = best_right_error;
        if (right == -1) {
            ((QueryData_Best *) root_pos_node->data)->test = best_right_class;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_error;
            ((QueryData_Best *) root_pos_node->data)->size = 1;
            ((QueryData_Best *) root_pos_node->data)->left = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_pos_node->data)->test = right;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_leafError;
            ((QueryData_Best *) root_pos_node->data)->size = 3;
        }

//        itemset.free();
//            cout << "cc0*" << endl;

        if (left != -1){
//                cout << "cc00" << endl;
            //insert left neg
            Array<Item > left_neg;
            left_neg.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 0), left_neg);
            TrieNode* left_neg_node = trie->insert(left_neg);
            left_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_neg_node->data)->error = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->leafError = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->test = best_left_class1;
            ((QueryData_Best *) left_neg_node->data)->size = 1;
            ((QueryData_Best *) left_neg_node->data)->left = nullptr;
            ((QueryData_Best *) left_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->left = (QueryData_Best *) left_neg_node->data;

            //insert left pos
            Array<Item > left_pos;
            left_pos.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 1), left_pos);
            TrieNode* left_pos_node = trie->insert(left_pos);
            left_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_pos_node->data)->error = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->leafError = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->test = best_left_class2;
            ((QueryData_Best *) left_pos_node->data)->size = 1;
            ((QueryData_Best *) left_pos_node->data)->left = nullptr;
            ((QueryData_Best *) left_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = (QueryData_Best *) left_pos_node->data;

            left_neg.free();
            left_pos.free();
            root_neg.free();
        }

        if (right != -1){
//                cout << "cc000" << endl;
            //insert right neg
            Array<Item > right_neg;
            right_neg.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 0), right_neg);
            TrieNode* right_neg_node = trie->insert(right_neg);
            right_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_neg_node->data)->error = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->leafError = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->test = best_right_class1;
            ((QueryData_Best *) right_neg_node->data)->size = 1;
            ((QueryData_Best *) right_neg_node->data)->left = nullptr;
            ((QueryData_Best *) right_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->left = (QueryData_Best *) right_neg_node->data;

            //insert right pos
            Array<Item > right_pos;
            right_pos.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 1), right_pos);
            TrieNode* right_pos_node = trie->insert(right_pos);
            right_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_pos_node->data)->error = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->leafError = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->test = best_right_class2;
            ((QueryData_Best *) right_pos_node->data)->size = 1;
            ((QueryData_Best *) right_pos_node->data)->left = nullptr;
            ((QueryData_Best *) right_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = (QueryData_Best *) right_pos_node->data;

            right_neg.free();
            right_pos.free();
            root_pos.free();
        }
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = best_root_error;
        ((QueryData_Best *) node->data)->leafError = root_leaf_error;
        ((QueryData_Best *) node->data)->test = root;
        ((QueryData_Best *) node->data)->size = ((QueryData_Best *) root_neg_node->data)->size + ((QueryData_Best *) root_pos_node->data)->size + 1;
        ((QueryData_Best *) node->data)->left = (QueryData_Best *) root_neg_node->data;
        ((QueryData_Best *) node->data)->right = (QueryData_Best *) root_pos_node->data;
//            cout << "cc1" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
        return node;
    }
    else{
        //error not lower than ub
//            cout << "cale" << endl;
        ErrorValues ev = query->computeErrorValues(cover);
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = FLT_MAX;
        ((QueryData_Best *) node->data)->leafError = ev.error;
        ((QueryData_Best *) node->data)->test = ev.maxclass;
        ((QueryData_Best *) node->data)->size = 1;
        ((QueryData_Best *) node->data)->left = nullptr;
        ((QueryData_Best *) node->data)->right = nullptr;
//            cout << "cc2" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
        return node;
    }
}*/



/*TrieNode* LcmPruned::getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb){
    ncall += 1;
    clock_t tt = clock();
    bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;

    Supports parent_sup_clas = copySupports(cover->getSupportPerClass());
    int parent_sup = cover->getSupport();

    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for (int m = 0; m < attributes_to_visit.size; ++m) {
        if (item_attribute(added) == attributes_to_visit[m]) continue;
        attr.push_back(attributes_to_visit[m]);
    }
    clock_t ttt = clock();
    Supports** sups = new Supports*[attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        sups[l] = new Supports[attr.size()];
        cover->intersect(attr[l]);
        sups[l][l] = cover->getSupportPerClass();
        for (int i = l+1; i < attr.size(); ++i) sups[l][i] = cover->intersectAndClass(attr[i]);
        cover->backtrack();
    }
    comptime += (clock() - ttt) / (float) CLOCKS_PER_SEC;


//    int root = -1, left = -1, right = -1;
//    float best_root_error = ub, best_left_error1 = FLT_MAX, best_left_error2 = FLT_MAX, best_right_error1 = FLT_MAX, best_right_error2 = FLT_MAX, best_left_error = FLT_MAX, best_right_error = FLT_MAX;
//    float root_leaf_error = query->computeErrorValues(cover).error, best_left_leafError = FLT_MAX, best_right_leafError = FLT_MAX;
//    int best_left_class1 = -1, best_left_class2 = -1, best_right_class1 = -1, best_right_class2 = -1, best_left_class = -1, best_right_class = -1;
    TwoTreeInfo treeinfo;

    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;
        if (item_attribute(added) == attr[i]){
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

//        float feat_left = -1, feat_right = -1;
//        float best_feat_left_error = FLT_MAX, best_feat_right_error = FLT_MAX, best_feat_left_error1 = FLT_MAX, best_feat_left_error2 = FLT_MAX, best_feat_right_error1 = FLT_MAX, best_feat_right_error2 = FLT_MAX;
//        float best_feat_left_leafError = FLT_MAX, best_feat_right_leafError = FLT_MAX;
//        int best_feat_left_class1 = -1, best_feat_left_class2 = -1, best_feat_right_class1 = -1, best_feat_right_class2 = -1, best_feat_left_class = -1, best_feat_right_class = -1;
        TwoTreeInfo best_i_tree;
        best_i_tree.root_error = ub;
        ErrorValues ev_left, ev_right;

        Supports idsc = sups[i][i];
        Support ids = sumSupports(idsc);
        Supports igsc = newSupports();
        subSupports(parent_sup_clas, idsc, igsc);
        Support igs = parent_sup - ids;

        //feature to left
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup){
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }
            // the feature at root cannot be splitted at left. It is then a leaf node
        else if (igs < 2 * query->minsup){
            ev_left = query->computeErrorValues(igsc);
            if (ev_left.error < best_i_tree.l_error || (ev_left.error == best_i_tree.l_error && best_i_tree.l_class > -1)){
                best_i_tree.l_error = ev_left.error;
                best_i_tree.l_class = ev_left.maxclass;
            }
            if (verbose) cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << ev_left.error << " on backtrack" << endl;
        }
            // the root node can theorically be split at left
        else {
            if (verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            ev_left = query->computeErrorValues(igsc);
            best_i_tree.l1_error = min(min(ev_left.error, best_i_tree.l_error), best_i_tree.root_error);
            Error tmp_left_leafError = ev_left.error;
            Error tmp_left_class = ev_left.maxclass;

            if (tmp_left_leafError != lb){
                ErrorValues ev;
                for (int j = 0; j < attr.size(); ++j) {
                    if (verbose) cout << "left test: " << attr[j] << endl;
                    if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
                        if (verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }
                    Supports jdsc = sups[j][j], idjdsc = sups[min(i,j)][max(i,j)], igjdsc = newSupports(); subSupports(jdsc, idjdsc, igjdsc);
                    Support jds = sumSupports(jdsc), idjds = sumSupports(idjdsc), igjds = sumSupports(igjdsc); Support igjgs = igs - igjds;

                    // the root node can in practice be split into two children
                    if (igjgs >= query->minsup && igjds >= query->minsup) {
                        if (verbose) cout << "le left testé peut splitter. on le regarde" << endl;

                        ev = query->computeErrorValues(igjdsc);
                        float tmp_left_error2 = ev.error;
                        int tmp_left_class2 = ev.maxclass;
                        if (verbose) cout << "le left a droite produit une erreur de " << tmp_left_error2 << endl;

                        if (tmp_left_error2 >= tmp_left_error) {
                            if (verbose) cout << "l'erreur droite du left montre rien de bon. Il est plus grand que le best so far... Un autre left..." << endl;
                            continue;
                        }

                        Supports igjgsc = newSupports(); subSupports(igsc, igjdsc, igjgsc);

                        ev = query->computeErrorValues(igjgsc);
                        float tmp_left_error1 = ev.error;
                        int tmp_left_class1 = ev.maxclass;
                        if (verbose) cout << "le left a gauche produit une erreur de " << tmp_left_error1 << endl;

                        if (tmp_left_error1 + tmp_left_error2 < tmp_left_error) {
                            best_i_tree.l_error = tmp_left_error1 + tmp_left_error2;
                            if (verbose) cout << "ce left ci donne une meilleure erreur que les précédents left: " << best_i_tree.l_error << endl;
                            best_i_tree.l1_error = tmp_left_error1;
                            best_i_tree.l2_error = tmp_left_error2;
                            best_i_tree.l1_class = tmp_left_class1;
                            best_i_tree.l1_class = tmp_left_class2;
                            best_i_tree.l_attr = attr[j];
                            if (best_i_tree.l_error == lb) break;
                        }
                        else if (verbose) cout << "l'erreur du left = " << tmp_left_error1 + tmp_left_error2 << " n'ameliore pas l'existant. Un autre left..." << endl;
                        deleteSupports(igjgsc);
                    }
                    else if (verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                    deleteSupports(igjdsc);
                }
                if (best_i_tree.l_error >= tmp_left_error && verbose) cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
            }
            else {
                if (verbose) cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
            }
        }


        //feature to right
        if (best_feat_left_error < best_root_error){
            if (verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * query->minsup){
                ErrorValues ev = query->computeErrorValues(idsc);
                best_feat_right_error = ev.error;
                best_feat_right_class = ev.maxclass;
                if (verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << best_feat_right_error << " on backtrack" << endl;
            }
            else {
                if (verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                ErrorValues ev = query->computeErrorValues(idsc);
                best_feat_right_error = min(ev.error, (best_root_error - best_feat_left_error));
                best_feat_right_leafError = ev.error;
                best_feat_right_class = ev.maxclass;
                float tmp = best_feat_right_error;
                if (ev.error != lb){
                    for (int j = 0; j < attr.size(); ++j) {
                        if (verbose) cout << "right test: " << attr[j] << endl;
                        if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
//                                cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }

                        Supports idjdsc = sups[min(i,j)][max(i,j)], idjgsc = newSupports(); subSupports(idsc, idjdsc, idjgsc);
                        int idjds = sumSupports(idjdsc), idjgs = sumSupports(idjgsc);

                        // the root node can in practice be split into two children
                        if (idjgs >= query->minsup && idjds >= query->minsup) {
                            if (verbose) cout << "le right testé peut splitter. on le regarde" << endl;
                            ev = query->computeErrorValues(idjgsc);
                            float tmp_right_error1 = ev.error;
                            int tmp_right_class1 = ev.maxclass;
                            if (verbose) cout << "le right a gauche produit une erreur de " << tmp_right_error1 << endl;

                            if (tmp_right_error1 >= min((best_root_error - best_feat_left_error), best_feat_right_error)){
                                if (verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                continue;
                            }

                            ev = query->computeErrorValues(idjdsc);
                            float tmp_right_error2 = ev.error;
                            int tmp_right_class2 = ev.maxclass;
                            if (verbose) cout << "le right a droite produit une erreur de " << tmp_right_error2 << endl;
                            if (tmp_right_error1 + tmp_right_error2 < min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                best_feat_right_error = tmp_right_error1 + tmp_right_error2;
                                if (verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << best_feat_right_error << endl;
                                best_feat_right_error1 = tmp_right_error1;
                                best_feat_right_error2 = tmp_right_error2;
                                best_feat_right_class1 = tmp_right_class1;
                                best_feat_right_class2 = tmp_right_class2;
                                feat_right = attr[j];
                                if (best_feat_right_error == lb) break;
                            }
                            else if (verbose) cout << "l'erreur du right = " << tmp_right_error1 + tmp_right_error2 << " n'ameliore pas l'existant. Un autre right..." << endl;
                        }
                        else if (verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        deleteSupports(idjgsc);
                    }
                    if (best_feat_right_error == tmp) if (verbose) cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: " << best_feat_right_error << endl;
                }
                else if (verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << best_feat_left_error << endl;
            }

            if (best_feat_left_error + best_feat_right_error < best_root_error){
                best_root_error = best_feat_left_error + best_feat_right_error;
                if (verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << best_root_error << endl;
                best_left_error = best_feat_left_error;
                best_right_error = best_feat_right_error;
                best_left_leafError = best_feat_left_leafError;
                best_right_leafError = best_feat_right_leafError;
                best_left_class = best_feat_left_class;
                best_right_class = best_feat_right_class;
                left = feat_left;
                right = feat_right;
                root = attr[i];
                best_left_error1 = best_feat_left_error1;
                best_left_error2 = best_feat_left_error2;
                best_right_error1 = best_feat_right_error1;
                best_right_error2 = best_feat_right_error2;
                best_left_class1 = best_feat_left_class1;
                best_left_class2 = best_feat_left_class2;
                best_right_class1 = best_feat_right_class1;
                best_right_class2 = best_feat_right_class2;
            }
        }
        deleteSupports(igsc);
    }
    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) {
            deleteSupports(sups[k][i]);
        }
    }
    if (verbose) cout << "root: " << root << " left: " << left << " right: " << right << endl;
    if (verbose) cout << "le1: " << best_left_error1 << " le2: " << best_left_error2 << " re1: " << best_right_error1 << " re2: " << best_right_error2 << endl;
    if (verbose) cout << "ble: " << best_left_error << " bre: " << best_right_error << " broe: " << best_root_error << endl;
    if (verbose) cout << "lc1: " << best_left_class1 << " lc2: " << best_left_class2 << " rc1: " << best_right_class1 << " rc2: " << best_right_class2 << endl;
    if (verbose) cout << "blc: " << best_left_class << " brc: " << best_right_class << endl;
//    cout << "temps find: " << (clock() - tt) / (float) CLOCKS_PER_SEC << " ";

    if (root != -1){
//            cout << "cc0" << endl;
        //insert root to left
        Array<Item > root_neg;
        root_neg.alloc(itemset.size + 1);
        addItem(itemset, item(root, 0), root_neg);
        TrieNode* root_neg_node = trie->insert(root_neg);
        root_neg_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_neg_node->data)->error = best_left_error;
        if (left == -1) {
            ((QueryData_Best *) root_neg_node->data)->test = best_left_class;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_error;
            ((QueryData_Best *) root_neg_node->data)->size = 1;
            ((QueryData_Best *) root_neg_node->data)->left = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_neg_node->data)->test = left;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_leafError;
            ((QueryData_Best *) root_neg_node->data)->size = 3;
        }
//            cout << "cc1*" << endl;

        //insert root to right
        Array<Item > root_pos;
        root_pos.alloc(itemset.size + 1);
        addItem(itemset, item(root, 1), root_pos);
        TrieNode* root_pos_node = trie->insert(root_pos);
        root_pos_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_pos_node->data)->error = best_right_error;
        if (right == -1) {
            ((QueryData_Best *) root_pos_node->data)->test = best_right_class;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_error;
            ((QueryData_Best *) root_pos_node->data)->size = 1;
            ((QueryData_Best *) root_pos_node->data)->left = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_pos_node->data)->test = right;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_leafError;
            ((QueryData_Best *) root_pos_node->data)->size = 3;
        }

//        itemset.free();
//            cout << "cc0*" << endl;

        if (left != -1){
//                cout << "cc00" << endl;
            //insert left neg
            Array<Item > left_neg;
            left_neg.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 0), left_neg);
            TrieNode* left_neg_node = trie->insert(left_neg);
            left_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_neg_node->data)->error = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->leafError = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->test = best_left_class1;
            ((QueryData_Best *) left_neg_node->data)->size = 1;
            ((QueryData_Best *) left_neg_node->data)->left = nullptr;
            ((QueryData_Best *) left_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->left = (QueryData_Best *) left_neg_node->data;

            //insert left pos
            Array<Item > left_pos;
            left_pos.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 1), left_pos);
            TrieNode* left_pos_node = trie->insert(left_pos);
            left_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_pos_node->data)->error = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->leafError = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->test = best_left_class2;
            ((QueryData_Best *) left_pos_node->data)->size = 1;
            ((QueryData_Best *) left_pos_node->data)->left = nullptr;
            ((QueryData_Best *) left_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = (QueryData_Best *) left_pos_node->data;

            left_neg.free();
            left_pos.free();
            root_neg.free();
        }

        if (right != -1){
//                cout << "cc000" << endl;
            //insert right neg
            Array<Item > right_neg;
            right_neg.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 0), right_neg);
            TrieNode* right_neg_node = trie->insert(right_neg);
            right_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_neg_node->data)->error = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->leafError = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->test = best_right_class1;
            ((QueryData_Best *) right_neg_node->data)->size = 1;
            ((QueryData_Best *) right_neg_node->data)->left = nullptr;
            ((QueryData_Best *) right_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->left = (QueryData_Best *) right_neg_node->data;

            //insert right pos
            Array<Item > right_pos;
            right_pos.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 1), right_pos);
            TrieNode* right_pos_node = trie->insert(right_pos);
            right_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_pos_node->data)->error = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->leafError = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->test = best_right_class2;
            ((QueryData_Best *) right_pos_node->data)->size = 1;
            ((QueryData_Best *) right_pos_node->data)->left = nullptr;
            ((QueryData_Best *) right_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = (QueryData_Best *) right_pos_node->data;

            right_neg.free();
            right_pos.free();
            root_pos.free();
        }
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = best_root_error;
        ((QueryData_Best *) node->data)->leafError = root_leaf_error;
        ((QueryData_Best *) node->data)->test = root;
        ((QueryData_Best *) node->data)->size = ((QueryData_Best *) root_neg_node->data)->size + ((QueryData_Best *) root_pos_node->data)->size + 1;
        ((QueryData_Best *) node->data)->left = (QueryData_Best *) root_neg_node->data;
        ((QueryData_Best *) node->data)->right = (QueryData_Best *) root_pos_node->data;
//            cout << "cc1" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
        return node;
    }
    else{
        //error not lower than ub
//            cout << "cale" << endl;
        ErrorValues ev = query->computeErrorValues(cover);
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = FLT_MAX;
        ((QueryData_Best *) node->data)->leafError = ev.error;
        ((QueryData_Best *) node->data)->test = ev.maxclass;
        ((QueryData_Best *) node->data)->size = 1;
        ((QueryData_Best *) node->data)->left = nullptr;
        ((QueryData_Best *) node->data)->right = nullptr;
//            cout << "cc2" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
        return node;
    }
}*/


//correct
TrieNode* LcmPruned::getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb){
    //cout << "\t\t lb = " << lb << endl;
    //lb = 0;
    // if (lb > 0) cout << "lb = " << lb << endl;
    if (ub <= lb){
        // cout << "cc" << endl;
        node->data = query->initData(cover);
        //((QueryData_Best *) node->data)->error = ((QueryData_Best *) node->data)->leafError;
        return node;
    }
    ncall += 1;
//    cout << "ncall: " << ncall;
    clock_t tt = clock();
    //cout << "tempss: " << (clock() - tt) / (float) CLOCKS_PER_SEC;
    bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;

    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    Support root_sup = cover->getSupport();
//    cout << "root_sup_class = " << root_sup_clas[0] << ", " << root_sup_clas[1] << endl;

    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for (int m = 0; m < attributes_to_visit.size; ++m) {
        if (item_attribute(added) == attributes_to_visit[m]) continue;
        attr.push_back(attributes_to_visit[m]);
    }
    clock_t ttt = clock();
    Supports** sups = new Supports*[attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        sups[l] = new Supports[attr.size()];
        cover->intersect(attr[l]);
        sups[l][l] = cover->getSupportPerClass();
        for (int i = l+1; i < attr.size(); ++i) sups[l][i] = cover->intersectAndClass(attr[i]);
        cover->backtrack();
    }
    comptime += (clock() - ttt) / (float) CLOCKS_PER_SEC;
//    cout << " temps comp: " << (clock() - ttt) / (float) CLOCKS_PER_SEC << " ";
//    exit(0);


    Attribute root = -1, left = -1, right = -1;
    Error best_root_error = ub, best_left_error1 = FLT_MAX, best_left_error2 = FLT_MAX, best_right_error1 = FLT_MAX, best_right_error2 = FLT_MAX, best_left_error = FLT_MAX, best_right_error = FLT_MAX;
    Supports best_root_corrects = nullptr, best_left_corrects1 = nullptr, best_left_corrects2 = nullptr, best_right_corrects1 = nullptr, best_right_corrects2 = nullptr, best_left_corrects = nullptr, best_right_corrects = nullptr ;
    Supports best_root_falses = nullptr, best_left_falses1 = nullptr, best_left_falses2 = nullptr, best_right_falses1 = nullptr, best_right_falses2 = nullptr, best_left_falses = nullptr, best_right_falses = nullptr ;
    Error root_leaf_error = query->computeErrorValues(cover).error, best_left_leafError = FLT_MAX, best_right_leafError = FLT_MAX;
    Class best_left_class1 = -1, best_left_class2 = -1, best_right_class1 = -1, best_right_class2 = -1, best_left_class = -1, best_right_class = -1;
    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;
        if (item_attribute(added) == attr[i]){
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

        Attribute feat_left = -1, feat_right = -1;
        Error best_feat_left_error = FLT_MAX, best_feat_right_error = FLT_MAX, best_feat_left_error1 = FLT_MAX, best_feat_left_error2 = FLT_MAX, best_feat_right_error1 = FLT_MAX, best_feat_right_error2 = FLT_MAX;
        Error best_feat_left_leafError = FLT_MAX, best_feat_right_leafError = FLT_MAX;
        Class best_feat_left_class1 = -1, best_feat_left_class2 = -1, best_feat_right_class1 = -1, best_feat_right_class2 = -1, best_feat_left_class = -1, best_feat_right_class = -1;
        Supports best_feat_root_corrects = nullptr, best_feat_left_corrects1 = nullptr, best_feat_left_corrects2 = nullptr, best_feat_right_corrects1 = nullptr, best_feat_right_corrects2 = nullptr, best_feat_left_corrects = nullptr, best_feat_right_corrects = nullptr ;
        Supports best_feat_root_falses = nullptr, best_feat_left_falses1 = nullptr, best_feat_left_falses2 = nullptr, best_feat_right_falses1 = nullptr, best_feat_right_falses2 = nullptr, best_feat_left_falses = nullptr, best_feat_right_falses = nullptr ;
        //int parent_sup = sumSupports(sups[i][i]);

        Supports idsc = sups[i][i];
        Support ids = sumSupports(idsc);
        Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc);
        Support igs = root_sup - ids;
//        cout << "idsc = " << idsc[0] << ", " << idsc[1] << endl;
//        cout << "igsc = " << igsc[0] << ", " << igsc[1] << endl;

        //feature to left
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup){
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }
            // the feature at root cannot be splitted at left. It is then a leaf node
        else if (igs < 2 * query->minsup){
            ErrorValues ev = query->computeErrorValues(igsc);
            best_feat_left_error = ev.error;
            best_feat_left_class = ev.maxclass;
            best_feat_left_corrects = ev.corrects;
            best_feat_left_falses = ev.falses;
            if (verbose) cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << best_feat_left_error << " on backtrack" << endl;
        }
            // the root node can theorically be split at left
        else {
            if (verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            ErrorValues ev = query->computeErrorValues(igsc);
            best_feat_left_error = min(ev.error, best_root_error);
            best_feat_left_leafError = ev.error;
            best_feat_left_class = ev.maxclass;
            best_feat_left_corrects = ev.corrects;
            best_feat_left_falses = ev.falses;
            if (ev.error != lb){
                Error tmp = best_feat_left_error;
                for (int j = 0; j < attr.size(); ++j) {
                    if (verbose) cout << "left test: " << attr[j] << endl;
                    if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
                        if (verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }
                    Supports jdsc = sups[j][j], idjdsc = sups[min(i,j)][max(i,j)], igjdsc = newSupports(); subSupports(jdsc, idjdsc, igjdsc);
                    Support jds = sumSupports(jdsc), idjds = sumSupports(idjdsc), igjds = sumSupports(igjdsc); Support igjgs = igs - igjds;

                    // the root node can in practice be split into two children
                    if (igjgs >= query->minsup && igjds >= query->minsup) {
                        if (verbose) cout << "le left testé peut splitter. on le regarde" << endl;

                        ev = query->computeErrorValues(igjdsc);
                        Error tmp_left_error2 = ev.error;
                        Class tmp_left_class2 = ev.maxclass;
                        Supports tmp_left_corrects2 = ev.corrects;
                        Supports tmp_left_falses2 = ev.falses;
                        if (verbose) cout << "le left a droite produit une erreur de " << tmp_left_error2 << endl;

                        if (tmp_left_error2 >= min(best_root_error, best_feat_left_error)) {
                            deleteSupports(tmp_left_corrects2); deleteSupports(tmp_left_falses2);
                            if (verbose) cout << "l'erreur gauche du left montre rien de bon. best root: " << best_root_error << " best left: " << best_feat_left_error << " Un autre left..." << endl;
                            continue;
                        }

                        Supports igjgsc = newSupports(); subSupports(igsc, igjdsc, igjgsc);
                        ev = query->computeErrorValues(igjgsc);
                        Error tmp_left_error1 = ev.error;
                        Class tmp_left_class1 = ev.maxclass;
                        Supports tmp_left_corrects1 = ev.corrects;
                        Supports tmp_left_falses1 = ev.falses;
                        if (verbose) cout << "le left a gauche produit une erreur de " << tmp_left_error1 << endl;

                        if (tmp_left_error1 + tmp_left_error2 < min(best_root_error, best_feat_left_error)) {
                            best_feat_left_error = tmp_left_error1 + tmp_left_error2;
                            plusSupports(tmp_left_corrects1, tmp_left_corrects2, best_feat_left_corrects);
                            plusSupports(tmp_left_falses1, tmp_left_falses2, best_feat_left_falses);
                            if (verbose) cout << "ce left ci donne une meilleure erreur que les précédents left: " << best_feat_left_error << endl;
                            best_feat_left_error1 = tmp_left_error1; best_feat_left_error2 = tmp_left_error2;
                            best_feat_left_class1 = tmp_left_class1; best_feat_left_class2 = tmp_left_class2;
                            if (feat_left != -1){
                                deleteSupports(best_feat_left_corrects1); deleteSupports(best_feat_left_corrects2);
                                deleteSupports(best_feat_left_falses1); deleteSupports(best_feat_left_falses2);
                            }
                            best_feat_left_corrects1 = tmp_left_corrects1; best_feat_left_corrects2 = tmp_left_corrects2;
                            best_feat_left_falses1 = tmp_left_falses1; best_feat_left_falses2 = tmp_left_falses2;
                            feat_left = attr[j];
                            if (best_feat_left_error == lb) break;
                        }
                        else {
                            deleteSupports(tmp_left_corrects1); deleteSupports(tmp_left_falses1);
                            deleteSupports(tmp_left_corrects2); deleteSupports(tmp_left_falses2);
                            if (verbose) cout << "l'erreur du left = " << tmp_left_error1 + tmp_left_error2 << " n'ameliore pas l'existant. Un autre left..." << endl;
                        }
                        deleteSupports(igjgsc);
                    }
                    else if (verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                    deleteSupports(igjdsc);
                }
                if (best_feat_left_error == tmp && verbose) cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
            }
            else {
                if (verbose) cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << best_feat_left_error << endl;
            }
        }


        //feature to right
        if (best_feat_left_error < best_root_error){
            if (verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * query->minsup){
                ErrorValues ev = query->computeErrorValues(idsc);
                best_feat_right_error = ev.error;
                best_feat_right_class = ev.maxclass;
                best_feat_right_corrects = ev.corrects;
                best_feat_right_falses = ev.falses;
                if (verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << best_feat_right_error << " on backtrack" << endl;
            }
            else {
                if (verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                ErrorValues ev = query->computeErrorValues(idsc);
                best_feat_right_error = min(ev.error, (best_root_error - best_feat_left_error));
                best_feat_right_leafError = ev.error;
                best_feat_right_class = ev.maxclass;
                best_feat_right_corrects = ev.corrects;
                best_feat_right_falses = ev.falses;
                Error tmp = best_feat_right_error;
                if (ev.error != lb){
                    for (int j = 0; j < attr.size(); ++j) {
                        if (verbose) cout << "right test: " << attr[j] << endl;
                        if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
                                if (verbose) cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }

                        Supports idjdsc = sups[min(i,j)][max(i,j)], idjgsc = newSupports(); subSupports(idsc, idjdsc, idjgsc);
                        Support idjds = sumSupports(idjdsc), idjgs = sumSupports(idjgsc);

                        // the root node can in practice be split into two children
                        if (idjgs >= query->minsup && idjds >= query->minsup) {
                            if (verbose) cout << "le right testé peut splitter. on le regarde" << endl;
                            ev = query->computeErrorValues(idjgsc);
                            Error tmp_right_error1 = ev.error;
                            Class tmp_right_class1 = ev.maxclass;
                            Supports tmp_right_corrects1 = ev.corrects;
                            Supports tmp_right_falses1 = ev.falses;
                            if (verbose) cout << "le right a gauche produit une erreur de " << tmp_right_error1 << endl;

                            if (tmp_right_error1 >= min((best_root_error - best_feat_left_error), best_feat_right_error)){
                                deleteSupports(tmp_right_corrects1); deleteSupports(tmp_right_falses1);
                                if (verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                continue;
                            }

                            ev = query->computeErrorValues(idjdsc);
                            Error tmp_right_error2 = ev.error;
                            Class tmp_right_class2 = ev.maxclass;
                            Supports tmp_right_corrects2 = ev.corrects;
                            Supports tmp_right_falses2 = ev.falses;
                            if (verbose) cout << "le right a droite produit une erreur de " << tmp_right_error2 << endl;
                            if (tmp_right_error1 + tmp_right_error2 < min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                best_feat_right_error = tmp_right_error1 + tmp_right_error2;
                                plusSupports(tmp_right_corrects1, tmp_right_corrects2, best_feat_right_corrects);
                                plusSupports(tmp_right_falses1, tmp_right_falses2, best_feat_right_falses);
                                if (verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << best_feat_right_error << endl;
                                best_feat_right_error1 = tmp_right_error1; best_feat_right_error2 = tmp_right_error2;
                                best_feat_right_class1 = tmp_right_class1; best_feat_right_class2 = tmp_right_class2;
                                if (feat_right != -1){
                                    deleteSupports(best_feat_right_corrects1); deleteSupports(best_feat_right_corrects2);
                                    deleteSupports(best_feat_right_falses1); deleteSupports(best_feat_right_falses2);
                                }
                                best_feat_right_corrects1 = tmp_right_corrects1; best_feat_right_corrects2 = tmp_right_corrects2;
                                best_feat_right_falses1 = tmp_right_falses1; best_feat_right_falses2 = tmp_right_falses2;
                                feat_right = attr[j];
                                if (best_feat_right_error == lb) break;
                            }
                            else {
                                deleteSupports(tmp_right_corrects1); deleteSupports(tmp_right_falses1);
                                deleteSupports(tmp_right_corrects2); deleteSupports(tmp_right_falses2);
                                if (verbose) cout << "l'erreur du right = " << tmp_right_error1 + tmp_right_error2 << " n'ameliore pas l'existant. Un autre right..." << endl;
                            }
                        }
                        else if (verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        deleteSupports(idjgsc);
                    }
                    if (best_feat_right_error == tmp) if (verbose) cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: " << best_feat_right_error << endl;
                }
                else if (verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << best_feat_left_error << endl;
            }

            if (best_feat_left_error + best_feat_right_error < best_root_error){
//                cout << "o1" << endl;
                best_root_error = best_feat_left_error + best_feat_right_error;
                if (verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << best_root_error << endl;
                best_left_error = best_feat_left_error; best_right_error = best_feat_right_error; best_left_leafError = best_feat_left_leafError; best_right_leafError = best_feat_right_leafError;
                best_left_class = best_feat_left_class; best_right_class = best_feat_right_class;
                left = feat_left; right = feat_right;
                root = attr[i];
                best_left_error1 = best_feat_left_error1; best_left_error2 = best_feat_left_error2; best_right_error1 = best_feat_right_error1; best_right_error2 = best_feat_right_error2;
                best_left_class1 = best_feat_left_class1; best_left_class2 = best_feat_left_class2; best_right_class1 = best_feat_right_class1; best_right_class2 = best_feat_right_class2;
                best_left_corrects = best_feat_left_corrects; best_left_falses = best_feat_left_falses;
                best_right_corrects = best_feat_right_corrects; best_right_falses = best_feat_right_falses;
                if (feat_left -= -1){
                    best_left_corrects1 = best_feat_left_corrects1; best_left_falses1 = best_feat_left_falses1;
                    best_left_corrects2 = best_feat_left_corrects2; best_left_falses2 = best_feat_left_falses2;
                }
                if (feat_right -= -1){
                    best_right_corrects1 = best_feat_right_corrects1; best_right_falses1 = best_feat_right_falses1;
                    best_right_corrects2 = best_feat_right_corrects2; best_right_falses2 = best_feat_right_falses2;
                }
            }
            else{
//                cout << "o2" << endl;
//                cout << "feat_left = " << feat_left << " and feat_right = " << feat_right << endl;
//                cout << best_left_corrects << endl;
//                if (best_left_corrects) deleteSupports(best_left_corrects);
//                if (best_left_falses) deleteSupports(best_left_falses);
//                if (best_right_corrects) deleteSupports(best_right_corrects);
//                if (best_right_falses) deleteSupports(best_right_falses);
                if (feat_left != -1){
                    if (best_feat_left_corrects1) deleteSupports(best_feat_left_corrects1);
                    if (best_feat_left_falses1) deleteSupports(best_feat_left_falses1);
                    if (best_feat_left_corrects2) deleteSupports(best_feat_left_corrects2);
                    if (best_feat_left_corrects2) deleteSupports(best_feat_left_falses2);
                }
                if (feat_right != -1){
                    if (best_feat_right_corrects1) deleteSupports(best_feat_right_corrects1);
                    if (best_feat_right_falses1) deleteSupports(best_feat_right_falses1);
                    if (best_feat_right_corrects2) deleteSupports(best_feat_right_corrects2);
                    if (best_feat_right_falses2) deleteSupports(best_feat_right_falses2);
                }
            }
        }
        deleteSupports(igsc);
    }
    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) {
            deleteSupports(sups[k][i]);
        }
    }
    if (verbose) cout << "root: " << root << " left: " << left << " right: " << right << endl;
    if (verbose) cout << "le1: " << best_left_error1 << " le2: " << best_left_error2 << " re1: " << best_right_error1 << " re2: " << best_right_error2 << endl;
    if (verbose) cout << "ble: " << best_left_error << " bre: " << best_right_error << " broe: " << best_root_error << endl;
    if (verbose) cout << "lc1: " << best_left_class1 << " lc2: " << best_left_class2 << " rc1: " << best_right_class1 << " rc2: " << best_right_class2 << endl;
    if (verbose) cout << "blc: " << best_left_class << " brc: " << best_right_class << endl;
//    cout << "temps find: " << (clock() - tt) / (float) CLOCKS_PER_SEC << " ";

    if (root != -1){
//            cout << "cc0" << endl;
        //insert root to left
        Array<Item > root_neg;
        root_neg.alloc(itemset.size + 1);
        addItem(itemset, item(root, 0), root_neg);
        TrieNode* root_neg_node = trie->insert(root_neg);
        root_neg_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_neg_node->data)->error = best_left_error;
        ((QueryData_Best *) root_neg_node->data)->corrects = best_left_corrects;
        ((QueryData_Best *) root_neg_node->data)->falses = best_left_falses;
        if (left == -1) {
            ((QueryData_Best *) root_neg_node->data)->test = best_left_class;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_error;
            ((QueryData_Best *) root_neg_node->data)->size = 1;
            ((QueryData_Best *) root_neg_node->data)->left = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_neg_node->data)->test = left;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_leafError;
            ((QueryData_Best *) root_neg_node->data)->size = 3;
        }
//            cout << "cc1*" << endl;

        //insert root to right
        Array<Item > root_pos;
        root_pos.alloc(itemset.size + 1);
        addItem(itemset, item(root, 1), root_pos);
        TrieNode* root_pos_node = trie->insert(root_pos);
        root_pos_node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) root_pos_node->data)->error = best_right_error;
        ((QueryData_Best *) root_pos_node->data)->corrects = best_right_corrects;
        ((QueryData_Best *) root_pos_node->data)->falses = best_right_falses;
        if (right == -1) {
            ((QueryData_Best *) root_pos_node->data)->test = best_right_class;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_error;
            ((QueryData_Best *) root_pos_node->data)->size = 1;
            ((QueryData_Best *) root_pos_node->data)->left = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = nullptr;
        }
        else {
            ((QueryData_Best *) root_pos_node->data)->test = right;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_leafError;
            ((QueryData_Best *) root_pos_node->data)->size = 3;
        }

//        itemset.free();
//            cout << "cc0*" << endl;

        if (left != -1){
//                cout << "cc00" << endl;
            //insert left neg
            Array<Item > left_neg;
            left_neg.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 0), left_neg);
            TrieNode* left_neg_node = trie->insert(left_neg);
            left_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_neg_node->data)->error = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->leafError = best_left_error1;
            ((QueryData_Best *) left_neg_node->data)->test = best_left_class1;
            ((QueryData_Best *) left_neg_node->data)->corrects = best_left_corrects1;
            ((QueryData_Best *) left_neg_node->data)->falses = best_left_falses1;
            ((QueryData_Best *) left_neg_node->data)->size = 1;
            ((QueryData_Best *) left_neg_node->data)->left = nullptr;
            ((QueryData_Best *) left_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->left = (QueryData_Best *) left_neg_node->data;

            //insert left pos
            Array<Item > left_pos;
            left_pos.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 1), left_pos);
            TrieNode* left_pos_node = trie->insert(left_pos);
            left_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) left_pos_node->data)->error = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->leafError = best_left_error2;
            ((QueryData_Best *) left_pos_node->data)->test = best_left_class2;
            ((QueryData_Best *) left_pos_node->data)->corrects = best_left_corrects2;
            ((QueryData_Best *) left_pos_node->data)->falses = best_left_falses2;
            ((QueryData_Best *) left_pos_node->data)->size = 1;
            ((QueryData_Best *) left_pos_node->data)->left = nullptr;
            ((QueryData_Best *) left_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_neg_node->data)->right = (QueryData_Best *) left_pos_node->data;

            left_neg.free();
            left_pos.free();
            root_neg.free();
        }

        if (right != -1){
//                cout << "cc000" << endl;
            //insert right neg
            Array<Item > right_neg;
            right_neg.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 0), right_neg);
            TrieNode* right_neg_node = trie->insert(right_neg);
            right_neg_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_neg_node->data)->error = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->leafError = best_right_error1;
            ((QueryData_Best *) right_neg_node->data)->test = best_right_class1;
            ((QueryData_Best *) right_neg_node->data)->corrects = best_right_corrects1;
            ((QueryData_Best *) right_neg_node->data)->falses = best_right_falses1;
            ((QueryData_Best *) right_neg_node->data)->size = 1;
            ((QueryData_Best *) right_neg_node->data)->left = nullptr;
            ((QueryData_Best *) right_neg_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->left = (QueryData_Best *) right_neg_node->data;

            //insert right pos
            Array<Item > right_pos;
            right_pos.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 1), right_pos);
            TrieNode* right_pos_node = trie->insert(right_pos);
            right_pos_node->data = (QueryData *) new QueryData_Best();
            ((QueryData_Best *) right_pos_node->data)->error = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->leafError = best_right_error2;
            ((QueryData_Best *) right_pos_node->data)->test = best_right_class2;
            ((QueryData_Best *) right_pos_node->data)->corrects = best_right_corrects2;
            ((QueryData_Best *) right_pos_node->data)->falses = best_right_falses2;
            ((QueryData_Best *) right_pos_node->data)->size = 1;
            ((QueryData_Best *) right_pos_node->data)->left = nullptr;
            ((QueryData_Best *) right_pos_node->data)->right = nullptr;
            ((QueryData_Best *) root_pos_node->data)->right = (QueryData_Best *) right_pos_node->data;

            right_neg.free();
            right_pos.free();
            root_pos.free();
        }
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = best_root_error;
        ((QueryData_Best *) node->data)->leafError = root_leaf_error;
        ((QueryData_Best *) node->data)->test = root;
        ((QueryData_Best *) node->data)->size = ((QueryData_Best *) root_neg_node->data)->size + ((QueryData_Best *) root_pos_node->data)->size + 1;
        ((QueryData_Best *) node->data)->left = (QueryData_Best *) root_neg_node->data;
        ((QueryData_Best *) node->data)->right = (QueryData_Best *) root_pos_node->data;
        ((QueryData_Best *) node->data)->corrects = newSupports(); plusSupports( ((QueryData_Best *) node->data)->left->corrects, ((QueryData_Best *) node->data)->right->corrects, ((QueryData_Best *) node->data)->corrects );
        ((QueryData_Best *) node->data)->falses = newSupports(); plusSupports( ((QueryData_Best *) node->data)->left->falses, ((QueryData_Best *) node->data)->right->falses, ((QueryData_Best *) node->data)->falses );

//            cout << "cc1" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
        return node;
    }
    else{
        //error not lower than ub
//            cout << "cale" << endl;
        ErrorValues ev = query->computeErrorValues(cover);
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = FLT_MAX;
        ((QueryData_Best *) node->data)->leafError = ev.error;
        ((QueryData_Best *) node->data)->corrects = ev.corrects;
        ((QueryData_Best *) node->data)->falses = ev.falses;
        ((QueryData_Best *) node->data)->test = ev.maxclass;
        ((QueryData_Best *) node->data)->size = 1;
        ((QueryData_Best *) node->data)->left = nullptr;
        ((QueryData_Best *) node->data)->right = nullptr;
//            cout << "cc2" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
        return node;
    }
}


void filldatawithinfo(QueryData_Best* data, ErrorValues ev, Attribute test = -1, Error lowerBound = NO_ERR, int size = 1){
    if (test != -1) data->test = test;
    else data->test = ev.maxclass;
    data->left = nullptr, data->right = nullptr;
    data->leafError = ev.error;
    data->error = FLT_MAX;
    data->lowerBound = lowerBound;
    data->size = size;
    data->solutionDepth = -1;
    data->corrects = ev.corrects, data->falses = ev.falses;
}

/*TrieNode* LcmPruned::getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb){
    ncall += 1;
    if (ub <= lb){
        node->data = query->initData(cover);
        return node;
    }
    lb = 0;

    clock_t tt = clock();
    bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;
    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    int root_sup = cover->getSupport();

    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for (int m = 0; m < attributes_to_visit.size; ++m) {
        if (item_attribute(added) == attributes_to_visit[m]) continue;
        attr.push_back(attributes_to_visit[m]);
    }
    clock_t ttt = clock();
    Supports** sups = new Supports*[attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        sups[l] = new Supports[attr.size()];
        cover->intersect(attr[l]);
        sups[l][l] = cover->getSupportPerClass();
        for (int i = l+1; i < attr.size(); ++i) sups[l][i] = cover->intersectAndClass(attr[i]);
        cover->backtrack();
    }
    comptime += (clock() - ttt) / (float) CLOCKS_PER_SEC;
//    cout << " temps comp: " << (clock() - ttt) / (float) CLOCKS_PER_SEC << " ";
//    exit(0);


    QueryData_Best* root_data = (QueryData_Best *)query->initData(cover);
    node->data = (QueryData *) root_data;
    //QueryData_Best* root_data = (QueryData_Best *)node->data;
    //filldatawithinfo(root_data, query->computeErrorValues(cover));
    if (verbose) cout << "erreur a priori = " << root_data->leafError << endl;

    if (root_data->leafError == lb){
        if (verbose) cout << "l'erreur est minimale sans descente" << endl;
        root_data->error = root_data->leafError;
        return node;
    }

    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;
        if (item_attribute(added) == attr[i]){
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

        Error feat_ubound = (root_data->left) ? root_data->error : ub;
        //Error feat_lbound = lb;
        QueryData_Best* feat_root_data = new QueryData_Best();
        filldatawithinfo(feat_root_data, query->computeErrorValues(cover), attr[i]);
        QueryData_Best* feat_left_data = new QueryData_Best();
        QueryData_Best* feat_right_data = new QueryData_Best();
        //int parent_sup = sumSupports(sups[i][i]);

        Supports idsc = sups[i][i];
        Support ids = sumSupports(idsc);
        Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc);
        Support igs = root_sup - ids;

        //feature to left
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup){
//            if (feat_root_data->leafError < root_data->error){
//                delete root_data;
//                root_data = feat_root_data; root_data->error = root_data->leafError;
//                delete feat_left_data; delete feat_right_data;
//            }
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }
            // the feature at root cannot be splitted at left. It is then a leaf node
        else if (igs < 2 * query->minsup){
            QueryData_Best* tmp_feat_left_data = new QueryData_Best();
            ErrorValues ev = query->computeErrorValues(igsc);
            filldatawithinfo(tmp_feat_left_data, ev);
            tmp_feat_left_data->error = tmp_feat_left_data->leafError;
            if (tmp_feat_left_data->error < min(min(feat_ubound, feat_left_data->error), feat_left_data->leafError) ){
                if (feat_left_data) {
                    delete feat_left_data;
                    feat_left_data = nullptr;
                }
                feat_left_data = tmp_feat_left_data;
            }
            else if (tmp_feat_left_data) {
                delete tmp_feat_left_data;
                tmp_feat_left_data = nullptr;
            }
            if (verbose) cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << feat_left_data->error << " on backtrack" << endl;
        }
            // the root node can theorically be split at left
        else {
            if (verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            ErrorValues ev = query->computeErrorValues(igsc);
            filldatawithinfo(feat_left_data, ev);
            if (verbose) cout << "root gauche erreur a priori = " << feat_left_data->leafError << endl;
            if (feat_left_data->leafError != lb){
                //float tmp = best_feat_left_error;
                for (int j = 0; j < attr.size(); ++j) {
                    if (verbose) cout << "left test: " << attr[j] << endl;
                    if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
                        if (verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }

                    Supports jdsc = sups[j][j], idjdsc = sups[min(i,j)][max(i,j)], igjdsc = newSupports(); subSupports(jdsc, idjdsc, igjdsc);
                    Support jds = sumSupports(jdsc), idjds = sumSupports(idjdsc), igjds = sumSupports(igjdsc); Support igjgs = igs - igjds;

                    // the root node can in practice be split into two children
                    if (igjgs >= query->minsup && igjds >= query->minsup) {
                        if (verbose) cout << "le left testé peut splitter. on le regarde" << endl;

                        QueryData_Best* tmp_feat_left_data = new QueryData_Best(); ev = query->computeErrorValues(igsc); filldatawithinfo(tmp_feat_left_data, ev, attr[j]);
                        QueryData_Best* tmp_feat_left2_data = new QueryData_Best(); ev = query->computeErrorValues(igjdsc); filldatawithinfo(tmp_feat_left2_data, ev);
                        tmp_feat_left2_data->error = tmp_feat_left2_data->leafError;

                        if (verbose) cout << "le left a droite produit une erreur de " << tmp_feat_left2_data->error << endl;

                        if ( tmp_feat_left2_data->error >= min(min(feat_ubound, feat_left_data->error), feat_left_data->leafError) ) {
                            if (tmp_feat_left2_data) {
                                delete tmp_feat_left2_data;
                                tmp_feat_left2_data = nullptr;
                            }
                            if (tmp_feat_left_data){
                                delete tmp_feat_left_data;
                                tmp_feat_left_data = nullptr;
                            }
                            if (verbose) cout << "l'erreur gauche du left montre rien de bon. best root: " << feat_ubound << " best left: " << min(feat_left_data->leafError, feat_left_data->error) << " Un autre left..." << endl;
                            continue;
                        }

                        Supports igjgsc = newSupports(); subSupports(igsc, igjdsc, igjgsc);
                        QueryData_Best* tmp_feat_left1_data = new QueryData_Best(); ev = query->computeErrorValues(igjgsc); filldatawithinfo(tmp_feat_left1_data, ev);
                        tmp_feat_left1_data->error = tmp_feat_left1_data->leafError;

                        if (verbose) cout << "le left a gauche produit une erreur de " << tmp_feat_left1_data->error << endl;

                        if (tmp_feat_left1_data->error + tmp_feat_left2_data->error < min(min(feat_ubound, feat_left_data->error), feat_left_data->leafError) ) {
                            tmp_feat_left_data->error = tmp_feat_left1_data->error + tmp_feat_left2_data->error;
                            tmp_feat_left_data->size = tmp_feat_left1_data->size + tmp_feat_left2_data->size + 1;
                            tmp_feat_left_data->left = tmp_feat_left1_data; tmp_feat_left_data->right = tmp_feat_left2_data;
                            if (feat_left_data->left) {
                                delete feat_left_data->left;
                                feat_left_data->left = nullptr;
                            }
                            if (feat_left_data->right) {
                                delete feat_left_data->right;
                                feat_left_data->right = nullptr;
                            }
                            if (feat_left_data) {
                                delete feat_left_data;
                                feat_left_data = nullptr;
                            }
                            feat_left_data = tmp_feat_left_data;
                            if (verbose) cout << "ce left ci donne une meilleure erreur que les précédents left: " << feat_left_data->error << endl;
                            if (feat_left_data->error == lb) {
                                if (verbose) cout << "erreur minimale...on break les autres siblings" << endl;
                                break;
                            }
                        }
                        else{
                            if (verbose) cout << "l'erreur du left = " << tmp_feat_left1_data->error + tmp_feat_left2_data->error << " n'ameliore pas l'existant. Un autre left..." << endl;
                            if (tmp_feat_left1_data) {
                                delete tmp_feat_left1_data;
                                tmp_feat_left1_data = nullptr;
                            }
                            if (tmp_feat_left2_data) {
                                delete tmp_feat_left2_data;
                                tmp_feat_left2_data = nullptr;
                            }
                            if (tmp_feat_left_data){
                                delete tmp_feat_left_data;
                                tmp_feat_left_data = nullptr;
                            }
                        }
                        deleteSupports(igjgsc);
                    }
                    else if (verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                    deleteSupports(igjdsc);
                }
                if (feat_left_data->error == FLT_MAX){
                    feat_left_data->error = feat_left_data->leafError;
                    if (verbose) cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: " << feat_left_data->error << endl;
                }
            }
            else {
                feat_left_data->error = feat_left_data->leafError;
                if (verbose) cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << feat_left_data->error << endl;
            }
        }


        //feature to right
        if (feat_left_data->error < feat_ubound){
            feat_ubound -= feat_left_data->error;
            //feat_lbound = max(0.f, feat_lbound - feat_left_data->error);
            if (verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * query->minsup){
                QueryData_Best* tmp_feat_right_data = new QueryData_Best();
                ErrorValues ev = query->computeErrorValues(idsc);
                filldatawithinfo(feat_right_data, ev);
                tmp_feat_right_data->error = tmp_feat_right_data->leafError;
                if (tmp_feat_right_data->error < min(min(feat_ubound, feat_right_data->error), feat_right_data->leafError) ){
                    if (feat_right_data) {
                        delete feat_right_data;
                        feat_right_data = nullptr;
                    }
                    feat_right_data = tmp_feat_right_data;
                }
                else if (tmp_feat_right_data) {
                    delete tmp_feat_right_data;
                    tmp_feat_right_data = nullptr;
                }
                if (verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << feat_right_data->error << " on backtrack" << endl;
            }
            else {
                if (verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                ErrorValues ev = query->computeErrorValues(idsc);
                filldatawithinfo(feat_right_data, ev);
                if (verbose) cout << "root droite erreur a priori = " << feat_right_data->leafError << endl;
                if (feat_right_data->leafError != lb){
                    for (int j = 0; j < attr.size(); ++j) {
                        if (verbose) cout << "right test: " << attr[j] << endl;
                        if (item_attribute(added) == attr[j] || attr[i] == attr[j]) {
                            if (verbose) cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }

                        Supports idjdsc = sups[min(i,j)][max(i,j)], idjgsc = newSupports(); subSupports(idsc, idjdsc, idjgsc);
                        int idjds = sumSupports(idjdsc), idjgs = sumSupports(idjgsc);

                        // the root node can in practice be split into two children
                        if (idjgs >= query->minsup && idjds >= query->minsup) {
                            if (verbose) cout << "le right testé peut splitter. on le regarde" << endl;

                            QueryData_Best* tmp_feat_right_data = new QueryData_Best(); ev = query->computeErrorValues(idsc); filldatawithinfo(tmp_feat_right_data, ev, attr[j]);
                            QueryData_Best* tmp_feat_right1_data = new QueryData_Best(); ev = query->computeErrorValues(idjgsc); filldatawithinfo(tmp_feat_right1_data, ev);
                            tmp_feat_right1_data->error = tmp_feat_right1_data->leafError;

                            if (verbose) cout << "le right a gauche produit une erreur de " << tmp_feat_right1_data->error << endl;

                            if (tmp_feat_right1_data->error >= min(min(feat_ubound, feat_right_data->error), feat_right_data->leafError) ){
                                if (tmp_feat_right1_data) {
                                    delete tmp_feat_right1_data;
                                    tmp_feat_right1_data = nullptr;
                                }
                                if (tmp_feat_right_data) {
                                    delete tmp_feat_right_data;
                                    tmp_feat_right_data = nullptr;
                                }
                                if (verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                continue;
                            }

                            QueryData_Best* tmp_feat_right2_data = new QueryData_Best(); ev = query->computeErrorValues(idjdsc); filldatawithinfo(tmp_feat_right2_data, ev);
                            tmp_feat_right2_data->error = tmp_feat_right2_data->leafError;

                            if (verbose) cout << "le right a droite produit une erreur de " << tmp_feat_right2_data->error << endl;
                            if (tmp_feat_right1_data->error + tmp_feat_right2_data->error < min(min(feat_ubound, feat_right_data->error), feat_right_data->leafError) ) {
                                tmp_feat_right_data->error = tmp_feat_right1_data->error + tmp_feat_right2_data->error;
                                tmp_feat_right_data->size = tmp_feat_right1_data->size + tmp_feat_right2_data->size + 1;
                                tmp_feat_right_data->left = tmp_feat_right1_data; tmp_feat_right_data->right = tmp_feat_right2_data;
                                if (feat_right_data->left) {
                                    delete feat_right_data->left;
                                    feat_right_data->left = nullptr;
                                }
                                if (feat_right_data->right) {
                                    delete feat_right_data->right;
                                    feat_right_data->right = nullptr;
                                }
                                if (feat_right_data) {
                                    delete feat_right_data;
                                    feat_right_data = nullptr;
                                }
                                feat_right_data = tmp_feat_right_data;
                                if (verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << feat_right_data->error << endl;
                                if (feat_right_data->error == lb) {
                                    if (verbose) cout << "erreur minimale...on break les autres siblings" << endl;
                                    break;
                                }
                            }
                            else{
                                if (verbose) cout << "l'erreur du right = " << tmp_feat_right1_data->error + tmp_feat_right2_data->error << " n'ameliore pas l'existant. Un autre right..." << endl;
                                if (tmp_feat_right1_data) {
                                    delete tmp_feat_right1_data;
                                    tmp_feat_right1_data = nullptr;
                                }
                                if (tmp_feat_right2_data) {
                                    delete tmp_feat_right2_data;
                                    tmp_feat_right2_data = nullptr;
                                }
                                if (tmp_feat_right_data) {
                                    delete tmp_feat_right_data;
                                    tmp_feat_right_data = nullptr;
                                }
                            }
                        }
                        else if (verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        deleteSupports(idjgsc);
                    }
                    if (feat_right_data->error == FLT_MAX){
                        feat_right_data->error = feat_right_data->leafError;
                        if (verbose) cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: " << feat_right_data->error << endl;
                    }
                }
                else{
                    feat_right_data->error = feat_right_data->leafError;
                    if (verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << feat_right_data->error << endl;
                }
            }
            if (feat_left_data->error + feat_right_data->error < min(ub, root_data->error)){
                feat_root_data->error = feat_left_data->error + feat_right_data->error;
                feat_root_data->size = feat_left_data->size + feat_right_data->size + 1;
                feat_root_data->left = feat_left_data; feat_root_data->right = feat_right_data;

                if ( ((QueryData_Best*)root_data)->left && ((QueryData_Best*)root_data)->left->left ) {
                    delete ((QueryData_Best*)root_data)->left->left;
                    ((QueryData_Best*)root_data)->left->left = nullptr;
                    delete ((QueryData_Best*)root_data)->left->right;
                    ((QueryData_Best*)root_data)->left->right = nullptr;
                }
                if ( ((QueryData_Best*)root_data)->right && ((QueryData_Best*)root_data)->right->left ) {
                    delete ((QueryData_Best*)root_data)->right->left;
                    ((QueryData_Best*)root_data)->right->left = nullptr;
                    delete ((QueryData_Best*)root_data)->right->right;
                    ((QueryData_Best*)root_data)->right->right = nullptr;
                }
                if (((QueryData_Best*)root_data)->left) {
                    delete ((QueryData_Best*)root_data)->left;
                    ((QueryData_Best*)root_data)->left = nullptr;
                    delete ((QueryData_Best*)root_data)->right;
                    ((QueryData_Best*)root_data)->right = nullptr;
                }

                if (node->data) {
                    delete node->data;
                    node->data = nullptr;
                    root_data = nullptr;
                }
                root_data = feat_root_data;
                node->data = (QueryData *) feat_root_data;
//                root_data = (QueryData_Best *) node->data;

                if (verbose) cout << feat_left_data->error << " " << feat_right_data->error << " " << root_data->error << endl;
                if (verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << root_data->error << endl;
            }
            else{
                if (feat_left_data->left){
                    delete feat_left_data->left;
                    feat_left_data->left = nullptr;
                    delete feat_left_data->right;
                    feat_left_data->right = nullptr;
                }
                if (feat_right_data->left){
                    delete feat_right_data->left;
                    feat_right_data->left = nullptr;
                    delete feat_right_data->right;
                    feat_right_data->right = nullptr;
                }
                if (feat_left_data) {
                    delete feat_left_data;
                    feat_left_data = nullptr;
                }
                if (feat_right_data) {
                    delete feat_right_data;
                    feat_right_data = nullptr;
                }
                if (feat_root_data) {
                    delete feat_root_data;
                    feat_root_data = nullptr;
                }
            }
        }
        else{
            if (feat_left_data->left){
                delete feat_left_data->left;
                feat_left_data->left = nullptr;
                delete feat_left_data->right;
                feat_left_data->right = nullptr;
            }
            if (feat_left_data) {
                delete feat_left_data;
                feat_left_data = nullptr;
            }
            if (feat_right_data) {
                delete feat_right_data;
                feat_right_data = nullptr;
            }
            if (feat_root_data) {
                delete feat_root_data;
                feat_root_data = nullptr;
            }
        }
        deleteSupports(igsc);
        if (root_data->error == lb) {
            if (verbose) cout << "ce i donne déjà la meilleure arbre possible" << endl;
            break;
        }
    }
    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) {
            deleteSupports(sups[k][i]);
        }
    }

    if (verbose) cout << "cc " << root_data->error << " " << ((root_data->left) ? root_data->left->error : -111) << " " << ((root_data->right) ? root_data->right->error : -111) << endl;

    if (root_data->left){
        Array<Item> root_neg; root_neg.alloc(itemset.size + 1);
        addItem(itemset, item(root_data->test, 0), root_neg);
        TrieNode* root_neg_node = trie->insert(root_neg);
        root_neg_node->data = (QueryData *) root_data->left;

        Array<Item> root_pos; root_pos.alloc(itemset.size + 1);
        addItem(itemset, item(root_data->test, 1), root_pos);
        TrieNode* root_pos_node = trie->insert(root_pos);
        root_pos_node->data = (QueryData *) root_data->right;

        if (root_data->left->left){
            Array<Item> left_neg; left_neg.alloc(root_neg.size + 1);
            addItem(root_neg, item(root_data->left->test, 0), left_neg);
            TrieNode* left_neg_node = trie->insert(left_neg);
            left_neg_node->data = (QueryData *) root_data->left->left;
            left_neg.free();

            Array<Item> left_pos; left_pos.alloc(root_neg.size + 1);
            addItem(root_neg, item(root_data->left->test, 1), left_pos);
            TrieNode* left_pos_node = trie->insert(left_pos);
            left_pos_node->data = (QueryData *) root_data->left->right;
            left_pos.free();
        }

        if (root_data->right->left){
            Array<Item> right_neg; right_neg.alloc(root_pos.size + 1);
            addItem(root_pos, item(root_data->right->test, 0), right_neg);
            TrieNode* right_neg_node = trie->insert(right_neg);
            right_neg_node->data = (QueryData *) root_data->right->left;
            right_neg.free();

            Array<Item> right_pos; right_pos.alloc(root_pos.size + 1);
            addItem(root_pos, item(root_data->right->test, 1), right_pos);
            TrieNode* right_pos_node = trie->insert(right_pos);
            right_pos_node->data = (QueryData *) root_data->right->right;
            right_pos.free();
        }

        root_neg.free(); root_pos.free();
    }


    if (verbose) cout << "root: " << ((root_data->left) ? root_data->test : -1) << " left: " << ((root_data->left && root_data->left->left) ? root_data->left->test : -1) << " right: " << ((root_data->right && root_data->right->left) ? root_data->right->test : -1) << endl;
    if (verbose) cout << "le1: " << ((root_data->left && root_data->left->left) ? root_data->left->left->error : -1)
                      << " le2: " << ((root_data->left && root_data->left->right) ? root_data->left->right->error : -1)
                      << " re1: " << ((root_data->right && root_data->right->left) ? root_data->right->left->error : -1)
                      << " re2: " << ((root_data->right && root_data->right->right) ? root_data->right->right->error : -1)
                      << endl;

    if (verbose) cout << "ble: " << ((root_data->left) ? root_data->left->error : -1)
                      << " bre: " << ((root_data->right) ? root_data->right->error : -1)
                      << " broe: " << ((root_data) ? root_data->error : -1)
                      << endl;

    if (verbose) cout << "lc1: " << ((root_data->left && root_data->left->left) ? root_data->left->left->test : -1)
                      << " lc2: " << ((root_data->left && root_data->left->right) ? root_data->left->right->test : -1)
                      << " rc1: " << ((root_data->right && root_data->right->left) ? root_data->right->left->test : -1)
                      << " rc2: " << ((root_data->right && root_data->right->right) ? root_data->right->right->test : -1)
                      << endl;

    if (verbose) cout << "blc: " << ((root_data->left && root_data->left->left) ? root_data->left->left->test : -1)
                      << " lc2: " << ((root_data->left && root_data->left->right) ? root_data->left->right->test : -1)
                      << " rc1: " << ((root_data->right && root_data->right->left) ? root_data->right->left->test : -1)
                      << " rc2: " << ((root_data->right && root_data->right->right) ? root_data->right->right->test : -1)
                      << endl;
    //if (verbose) cout << "blc: " << best_left_class << " brc: " << best_right_class << endl;
//    cout << "temps find: " << (clock() - tt) / (float) CLOCKS_PER_SEC << " ";
    spectime += (clock() - tt) / (float) CLOCKS_PER_SEC;
    return node;
}*/

