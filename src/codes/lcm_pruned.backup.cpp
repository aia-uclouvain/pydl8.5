#include "lcm_pruned.h"
#include "query_best.h" // if cannot link is specified, we need a clustering problem!!!
#include "logger.h"
#include <iostream>
#include <limits.h>
#include <cassert>
#include <cmath>
#include "dataContinuous.h"
#include <chrono>

using namespace std::chrono;


/*struct Hash {
    size_t operator()(const int &vec) const {
        return vec;
    }
};*/

LcmPruned::LcmPruned(DataManager *dataReader, Query *query, Trie *trie, bool infoGain, bool infoAsc, bool repeatSort) :
        dataReader(dataReader), trie(trie), query(query), infoGain(infoGain), infoAsc(infoAsc), repeatSort(repeatSort) {
}

TrieNode *existingsolution(TrieNode *node, Error *nodeError) {
    Logger::showMessageAndReturn("the solution exists and it is worth : ", *nodeError);
    return node;
}

TrieNode *cannotsplitmore(TrieNode *node, Error ub, Error *nodeError, Error leafError) {
    Logger::showMessageAndReturn("max depth reached. ub = ", ub, " and leaf error = ", leafError);
    // we return the leaf error as node error without checking the upperbound constraint. The parent will do it
    *nodeError = leafError;
    return node;
}

TrieNode *reachlowest(TrieNode *node, Error *nodeError, Error leafError) {
    *nodeError = leafError;
    Logger::showMessageAndReturn("lowest error. node error = leaf error = ", *nodeError);
    return node;
}

TrieNode *infeasiblecase(TrieNode *node, Error *saved_lb, Error ub) {
    Logger::showMessageAndReturn("no solution bcoz ub < lb. lb =", *saved_lb, " and ub = ", ub);
    return node;
}

LcmPruned::~LcmPruned() {
}

/** recurse - this method finds the best tree given an itemset and its cover and update
 * the information of the node representing the itemset
 *
 * @param itemset - the itemset for which we are looking for the best tree
 * @param last_added - the last added attribute
 * @param node - the node representing the itemset
 * @param next_candidates - next attributes to visit
 * @param cover - the transactions covered by the itemset
 * @param depth - the current depth in the search tree
 * @param ub - the upper bound of the search. It cannot be reached
 * @param computed_lb - a computed similarity lower bound. It can be reached
 * @return the same node as get in parameter with added information about the best tree
 */
TrieNode *LcmPruned::recurse(Array<Item> itemset,
                             Attribute last_added,
                             TrieNode *node,
                             Array<Attribute> next_candidates,
                             RCover *cover,
                             Depth depth,
                             float ub,
                             float computed_lb) {

    // check if we ran out of time
    if (query->timeLimit > 0) {
        float runtime = duration_cast<milliseconds>(high_resolution_clock::now() - query->startTime).count() / 1000.0;
        if (runtime >= query->timeLimit)
            query->timeLimitReached = true;
    }

    // the node already exists
    if (node->data) {
        Logger::showMessageAndReturn("the node exists");

        Error *nodeError = &(((QueryData_Best *) node->data)->error);

        // in case the solution exists
        if (*nodeError < FLT_MAX) {
            return existingsolution(node, nodeError);
        }

        Error *saved_lb = &(((QueryData_Best *) node->data)->lowerBound);
        if (computed_lb > *saved_lb) *saved_lb = computed_lb;

        // in case the problem is infeasible
        if (ub <= *saved_lb) {
            return infeasiblecase(node, saved_lb, ub);
        }

        Error leafError = ((QueryData_Best *) node->data)->leafError;

        // we reach the lowest value possible. implicitely, the upper bound constraint is not violated
        if (floatEqual(leafError, *saved_lb)) {
            return reachlowest(node, nodeError, leafError);
        }

        // we cannot split more
        if (depth == query->maxdepth || cover->getSupport() < 2 * query->minsup) {
            return cannotsplitmore(node, ub, nodeError, leafError);
        }
    }

    // in case the solution cannot be derived without computation and remaining depth is 2, we use a specific algorithm
    if (query->maxdepth - depth == 2 && cover->getSupport() >= 2 * query->minsup) {
        // if the node exists, use the max between the saved lb and the similarity computed lb
        computed_lb = (node->data) ? max(computed_lb, ((QueryData_Best *) node->data)->lowerBound) : computed_lb;
        return getdepthtwotrees(cover, ub, next_candidates, last_added, itemset, node, computed_lb);
    }

    // there are two cases in which the execution attempt here
    // 1- when the node did not exist
    // 2- when the node exists without solution and its upper bound is higher than its lower bound


    // at this stage, we will probably make a search, so we create empty array. It will be replaced
    // by the good one after calling getSuccessors function
    Array<Attribute> next_attributes;

    // case 1 : the node did not exist
    // no need to create it. It was just created before the call to this function. we will just create it data information
    if (!node->data) {
        Logger::showMessageAndReturn("New node");
        latticesize++;

        // Initialize data information about the node and get them for the search
        node->data = query->initData(cover);
        Error *saved_lb = &(((QueryData_Best *) node->data)->lowerBound);
        if (computed_lb > *saved_lb) *saved_lb = computed_lb;
        Error leafError = ((QueryData_Best *) node->data)->leafError;
        Error *nodeError = &(((QueryData_Best *) node->data)->error);
        Logger::showMessageAndReturn("after init of the new node. ub = ", ub, " and leaf error = ", leafError);


        // in case the problem is infeasible
        if (ub <= *saved_lb) {
            return infeasiblecase(node, saved_lb, ub);
        }

        // we reach the lowest value possible. implicitely, the upper bound constraint is not violated
        // at this stage, all solution parameters have already been stored by initData apart from node error
        if (floatEqual(leafError, *saved_lb)) {
            return reachlowest(node, nodeError, leafError);
        }

        // we cannot split more
        if (depth == query->maxdepth || cover->getSupport() < 2 * query->minsup) {
            return cannotsplitmore(node, ub, nodeError, leafError);
        }

        // if time limit is reached we backtrack
        if (query->timeLimitReached) {
            *nodeError = leafError;
            return node;
        }

        // if we can't get solution without computation, we compute the next candidates to perform the search
        next_attributes = getSuccessors(next_candidates, cover, last_added);


    }
    //case 2 : the node exists without solution but ub > last ub which is now lb
    else {

        // update the lower bound if needed
        Error *saved_lb = &(((QueryData_Best *) node->data)->lowerBound);
        if (computed_lb > *saved_lb) *saved_lb = computed_lb;

        Error leafError = ((QueryData_Best *) node->data)->leafError;
        Error *nodeError = &(((QueryData_Best *) node->data)->error);
        Logger::showMessageAndReturn("existing node without solution and higher bound. leaf error = ", leafError,
                                     " new ub = ", ub);

        if (query->timeLimitReached) {
            *nodeError = leafError;
            return node;
        }

        // if we can't get solution without computation, we compute the next candidates to perform the search
        // next_attributes = getSuccessors(attributes_to_visit, cover, added, getExistingSuccessors(node));
        next_attributes = getSuccessors(next_candidates, cover, last_added);
        //<====================================  END STEP  ==========================================>
    }

    Error *lb = &(((QueryData_Best *) node->data)->lowerBound);
    Error leafError = ((QueryData_Best *) node->data)->leafError;
    Error *nodeError = &(((QueryData_Best *) node->data)->error);

    if (next_attributes.size == 0) {
        Logger::showMessageAndReturn("No candidates. nodeError is set to leafError");
        *nodeError = leafError;
        Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ",
                                     *nodeError);
        Logger::showMessageAndReturn("we backtrack");
        next_attributes.free();
        return node;
    }

    Error child_ub = ub;
    Error minlb = FLT_MAX, sumbound = 0;
    bool first = false;
    bitset<M> *covlb1 = nullptr;
    Supports sclb1 = nullptr;
    Supports sflb1 = nullptr;
    Error errlb1 = 0;
    bitset<M> *covlb2 = nullptr;
    Supports sclb2 = nullptr;
    Supports sflb2 = nullptr;
    Error errlb2 = FLT_MAX;
    bitset<M> *covlb3 = nullptr;
    Supports sclb3 = nullptr;
    Supports sflb3 = nullptr;
    Support suplb = 0;
    Error errlb3;
    forEach (i, next_attributes) {

//                Attribute last_attrs[2];
        Array<Item> itemsets[2];
        TrieNode *nodes[2];

        cover->intersect(next_attributes[i], false);
        Error llb = computeLowerBound(cover, covlb1, covlb2, covlb3, sclb1, sclb2, sclb3, sflb1, sflb2, sflb3);
//                items[0] = item(next_attributes[i], 0);
        itemsets[0] = addItem(itemset, item(next_attributes[i], 0));
        nodes[0] = trie->insert(itemsets[0]);
        llb = (nodes[0]->data) ? max(((QueryData_Best *) nodes[0]->data)->lowerBound, llb) : llb;
        //itemsets[0].free();
        cover->backtrack();

        cover->intersect(next_attributes[i]);
        Error rlb = computeLowerBound(cover, covlb1, covlb2, covlb3, sclb1, sclb2, sclb3, sflb1, sflb2, sflb3);
//                items[1] = item(next_attributes[i], 1);
        itemsets[1] = addItem(itemset, item(next_attributes[i], 1));
        nodes[1] = trie->insert(itemsets[1]);
        rlb = (nodes[1]->data) ? max(((QueryData_Best *) nodes[1]->data)->lowerBound, rlb) : rlb;
        //itemsets[1].free();
        cover->backtrack();
        sumbound = llb + rlb;
        first = rlb > llb;



        /*cover->intersect(next_attributes[i], first);
        Error llb = computeLowerBound(cover, covlb1, covlb2, covlb3, sclb1, sclb2, sclb3, sflb1, sflb2, sflb3);
        //cout << "llb : " << llb << endl;
        Item left_item = item(next_attributes[i], first);
        Array<Item> left_itemset = addItem(itemset, left_item);
        TrieNode *left_node = trie->insert(left_itemset);
        llb = (left_node->data) ? max( ((QueryData_Best *) left_node->data)->lowerBound, llb ) : llb;
        left_node = recurse(left_itemset, left_item, left_node, next_attributes, cover, depth + 1, child_ub, llb);
        addInfoForLowerBound(cover, left_node->data, errlb1, errlb2, errlb3, covlb1, covlb2, covlb3, sclb1, sclb2, sclb3, sflb1, sflb2, sflb3, suplb);
        Error leftError = ((QueryData_Best *) left_node->data)->error;
        left_itemset.free();
        cover->backtrack();*/

        cover->intersect(next_attributes[i], first);
        nodes[first] = recurse(itemsets[first], next_attributes[i], nodes[first], next_attributes, cover, depth + 1,
                               child_ub, llb);
        addInfoForLowerBound(cover, nodes[first]->data, errlb1, errlb2, errlb3, covlb1, covlb2, covlb3, sclb1,
                             sclb2, sclb3, sflb1, sflb2, sflb3, suplb);
        Error leftError = ((QueryData_Best *) nodes[first]->data)->error;
        itemsets[first].free();
        cover->backtrack();

        //if (query->canimprove(left_node->data, child_ub)) {
        if (query->canimprove(nodes[first]->data, child_ub)) {

            /*cover->intersect(next_attributes[i], !first);
            Error rlb = computeLowerBound(cover, covlb1, covlb2, covlb3, sclb1, sclb2, sclb3, sflb1, sflb2, sflb3);
            //cout << "rlb : " << rlb << endl;
            Item right_item = item(next_attributes[i], !first);
            Array<Item> right_itemset = addItem(itemset, right_item);
            TrieNode *right_node = trie->insert(right_itemset);
            rlb = (right_node->data) ? max( ((QueryData_Best *) right_node->data)->lowerBound, rlb ) : rlb;
            float remainUb = child_ub - leftError;
            right_node = recurse(right_itemset, right_item, right_node, next_attributes, cover, depth + 1, remainUb, rlb);
            addInfoForLowerBound(cover, right_node->data, errlb1, errlb2, errlb3, covlb1, covlb2, covlb3, sclb1, sclb2, sclb3, sflb1, sflb2, sflb3, suplb);
            Error rightError = ((QueryData_Best *) right_node->data)->error;
            right_itemset.free();
            cover->backtrack();*/


            cover->intersect(next_attributes[i], !first);
            float remainUb = child_ub - leftError;
            nodes[!first] = recurse(itemsets[!first], next_attributes[i], nodes[!first], next_attributes, cover,
                                    depth + 1, remainUb, rlb);
            addInfoForLowerBound(cover, nodes[!first]->data, errlb1, errlb2, errlb3, covlb1, covlb2, covlb3, sclb1,
                                 sclb2, sclb3, sflb1, sflb2, sflb3, suplb);
            Error rightError = ((QueryData_Best *) nodes[!first]->data)->error;
            itemsets[!first].free();
            cover->backtrack();

            Error feature_error = leftError + rightError;
            //bool hasUpdated = query->updateData(node->data, child_ub, next_attributes[i], left_node->data, right_node->data);
            bool hasUpdated = query->updateData(node->data, child_ub, next_attributes[i], nodes[first]->data,
                                                nodes[!first]->data);
            if (hasUpdated) child_ub = feature_error;
            if (hasUpdated)
                Logger::showMessageAndReturn("après cet attribut, node error = ", *nodeError, " et ub = ",
                                             child_ub);

            if (query->canSkip(node->data)) {//lowerBound reached
                Logger::showMessageAndReturn("C'est le meilleur. on break le reste");
                break; //prune remaining attributes not browsed yet
            }
        } else {
            if (sumbound < minlb) minlb = sumbound;
        }

        if (query->stopAfterError) {
            if (depth == 0 && ub < FLT_MAX) {
                if (*nodeError < ub)
                    break;
            }
        }
    }
    delete[] covlb1;
    delete[] covlb2;
    delete[] covlb3;

    if (floatEqual(*nodeError, FLT_MAX) && max(ub, minlb) > *lb) { //cache successors if solution not found
        //cout << "minlb = " << minlb << " " << *lb << endl;
        *lb = max(ub, minlb);
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


Error LcmPruned::computeLowerBound(RCover *cover, bitset<M> *covlb1, bitset<M> *covlb2, bitset<M> *covlb3,
                                   Supports sclb1, Supports sclb2, Supports sclb3,
                                   Supports sflb1, Supports sflb2, Supports sflb3) {
    Error tmp1 = 0, tmp2 = 0, tmp3 = 0;
    if (covlb1) {
        Error c = 0, f = 0;
        SupportClass nullerr = 0;
        Supports dif1 = cover->minusMe(covlb1);
        forEachClass(n) {
            c += max(nullerr, sclb1[n] - dif1[n]);
            f += max(nullerr, sflb1[n] - dif1[n]);
        }
        tmp1 = min(c, f);
        deleteSupports(dif1);
    }
    /*if (covlb2){
        Error c = 0, f = 0;
        Supports dif2 = cover->minusMe(covlb2);
        forEachClass(n){
            c += max(0, sclb2[n] - dif2[n]);
            f += max(0, sflb2[n] - dif2[n]);
        }
        tmp2 = min(c, f);
        deleteSupports(dif2);
    }*/
    if (covlb3) {
        Error c = 0, f = 0;
        SupportClass nullerr = 0;
        Supports dif3 = cover->minusMe(covlb3);
        forEachClass(n) {
            c += max(nullerr, sclb3[n] - dif3[n]);
            f += max(nullerr, sflb3[n] - dif3[n]);
        }
        tmp3 = min(c, f);
        deleteSupports(dif3);
    }
    //cout << "tmp1 : " << tmp1 << " et tmp2 : " << tmp2 << endl;
    return max(tmp1, tmp3);
}

void LcmPruned::addInfoForLowerBound(RCover *cover, QueryData *node_data, Error errlb1, Error errlb2, Error errlb3,
                                     bitset<M> *&covlb1, bitset<M> *&covlb2, bitset<M> *&covlb3,
                                     Supports &sclb1, Supports &sclb2, Supports &sclb3,
                                     Supports &sflb1, Supports &sflb2, Supports &sflb3,
                                     Support suplb) {
    if (((QueryData_Best *) node_data)->error < FLT_MAX) {
        Error b = (((QueryData_Best *) node_data)->error < FLT_MAX) ? ((QueryData_Best *) node_data)->error
                                                                    : ((QueryData_Best *) node_data)->lowerBound;
        if (b < FLT_MAX && b > errlb1) {
            delete[] covlb1;
            covlb1 = cover->getTopBitsetArray();
            sclb1 = ((QueryData_Best *) node_data)->corrects;
            sflb1 = ((QueryData_Best *) node_data)->falses;
            errlb1 = b;
        }
        /*if (b > 0 && b < errlb2){
            delete [] covlb2; covlb2 = cover->getTopBitsetArray();
            sclb2 = ((QueryData_Best *) node_data)->corrects;
            sflb2 = ((QueryData_Best *) node_data)->falses;
            errlb2 = b;
        }*/
        if (cover->getSupport() > suplb) {
            delete[] covlb3;
            covlb3 = cover->getTopBitsetArray();
            sclb3 = ((QueryData_Best *) node_data)->corrects;
            sflb3 = ((QueryData_Best *) node_data)->falses;
            suplb = cover->getSupport();
            errlb3 = b;
        }
    }
}


void LcmPruned::run() {
    query->setStartTime();
    // set the correct maxerror if needed
    float maxError = NO_ERR;
    if (query->maxError > 0) maxError = query->maxError;

    // Initialize the cover object. It represents a mask of covered transactions. At this beginning all the transactions are set to 1
    RCover *cover = new RCover(dataReader);

    // Create empty list for candidate attributes
    Array<Attribute> attributes_to_visit(nattributes, 0);

    // Update the candidate list based on frequency criterion
    if (query->minsup == 1) { // do not check frequency if minsup = 1
        for (int attr = 0; attr < nattributes; ++attr)
            attributes_to_visit.push_back(attr);
    } else { // make sure each candidate attribute can be split into two nodes fulfilling the frequency criterion
        for (int attr = 0; attr < nattributes; ++attr) {
            if (cover->intersectAndSup(attr, false) >= query->minsup && cover->intersectAndSup(attr) >= query->minsup)
                attributes_to_visit.push_back(attr);
        }
    }

    //create an empty array of items representing an empty set and insert it
    Array<Item> itemset;
    itemset.size = 0;
    itemset.elts = nullptr;
    TrieNode *node = trie->insert(itemset);

    // call the recursive function to start the search
    query->realroot = recurse(itemset, NO_ATTRIBUTE, node, attributes_to_visit, cover, 0, maxError);

    // never forget to return back what is not yours. Think to others who need it ;-)
    itemset.free();
    attributes_to_visit.free();
    delete cover;

    cout << "ncall: " << ncall << endl;
    cout << "spectime: " << spectime << endl;
    cout << "comptime: " << comptime << endl;
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


Array<Attribute> LcmPruned::getSuccessors(Array<Attribute> last_candidates, RCover *cover, Attribute last_added,
                                          unordered_set<int> frequent_attr) {

    std::multimap<float, Attribute> gain;
    Array<Attribute> next_candidates(last_candidates.size, 0);

    // the current node does not fullfill the frequency criterion. In correct situation, this case won't happen
    if (cover->getSupport() < 2 * query->minsup)
        return next_candidates;

    int current_sup = cover->getSupport();
    Supports current_sup_class = copySupports(cover->getClassSupport());

    // access each candidate
    forEach (i, last_candidates) {

        // this attribute is already in the current itemset
        if (last_added == last_candidates[i]) continue;

        // compute the support of each candidate
        int sup_left = cover->intersectAndSup(last_candidates[i], false);
        int sup_right = current_sup - sup_left; //no need to intersect with negative item to compute its support

        // add frequent attributes but if heuristic is used to sort them, compute its value and sort later
        if (sup_left >= query->minsup && sup_right >= query->minsup) {
            //continuous dataset. Not supported yet
            if (query->continuous) {}
            else {
                if (infoGain) {
                    // compute the support per class in each split of the attribute to compute its IG value
                    Supports sup_class_left = cover->intersectAndClass(last_candidates[i], false);
                    Supports sup_class_right = newSupports();
                    subSupports(current_sup_class, sup_class_left, sup_class_right);
                    gain.insert(std::pair<float, Attribute>(informationGain(sup_class_left, sup_class_right),
                                                            last_candidates[i]));
                    deleteSupports(sup_class_left);
                    deleteSupports(sup_class_right);
                } else next_candidates.push_back(last_candidates[i]);
            }
        }
    }
    deleteSupports(current_sup_class);

    // if heuristic is used, add the next candidates given the heuristic order
    if (infoGain) {
        if (infoAsc)
            for (multimap<float, int>::iterator it = gain.begin(); it != gain.end(); ++it)
                next_candidates.push_back(it->second); //items with low IG first
        else
            for (multimap<float, int>::reverse_iterator it = gain.rbegin(); it != gain.rend(); ++it)
                next_candidates.push_back(it->second); //items with high IG first
    }
    // disable the heuristic variable if the sort must be performed once
    if (!repeatSort) infoGain = false;

    return next_candidates;
}

Array<Attribute> LcmPruned::getExistingSuccessors(TrieNode *node) {
    // use an hashset to reduce the insert time. a basic int hasher is ok
    unordered_set<int> candidates_checker;
    int size = candidates_checker.size();
    Array<Attribute> candidates(node->edges.size(), 0);
    for (TrieEdge edge : node->edges) {
        candidates_checker.insert(item_attribute(edge.item));
        if (candidates_checker.size() > size) {
            candidates.push_back(item_attribute(edge.item));
            size++;
        }
    }
    return candidates;
}



/*TrieNode* LcmPruned::getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb){
    if (ub <= lb){ node->data = query->initData(cover); return node; }

    ncall += 1; bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;

    clock_t tt = clock();
    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    Support root_sup = cover->getSupport();
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


    Attribute root = -1, left = -1, right = -1;
    Error best_root_error = FLT_MAX;
    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;
        if (item_attribute(added) == attr[i]){
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

        Attribute feat_left = -1, feat_right = -1;
        Error best_feat_left_error = FLT_MAX, best_feat_right_error = FLT_MAX;

        Supports idsc = sups[i][i]; Support ids = sumSupports(idsc); Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc); Support igs = root_sup - ids;

        //feature to left
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup){
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }
            // the feature at root cannot be splitted at left. It is then a leaf node
        if (igs < 2 * query->minsup){
            best_feat_left_error = query->computeOnlyError(igsc);
            if (verbose) cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << best_feat_left_error << " on backtrack" << endl;
        }
            // the root node can theorically be split at left
        else {
            if (verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            best_feat_left_error = query->computeOnlyError(igsc);
            if (best_feat_left_error != lb){
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

                        Error tmp_left_error2 = query->computeOnlyError(igjdsc);
                        if (verbose) cout << "le left a droite produit une erreur de " << tmp_left_error2 << endl;

                        if (tmp_left_error2 >= min(best_root_error, best_feat_left_error)) {
                            if (verbose) cout << "l'erreur gauche du left montre rien de bon. best root: " << best_root_error << " best left: " << best_feat_left_error << " Un autre left..." << endl;
                            continue;
                        }

                        Supports igjgsc = newSupports(); subSupports(igsc, igjdsc, igjgsc);
                        Error tmp_left_error1 = query->computeOnlyError(igjgsc);
                        if (verbose) cout << "le left a gauche produit une erreur de " << tmp_left_error1 << endl;

                        if (tmp_left_error1 + tmp_left_error2 < min(best_root_error, best_feat_left_error)) {
                            best_feat_left_error = tmp_left_error1 + tmp_left_error2;
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
}*/


//correct
TrieNode *
LcmPruned::getdepthtwotrees(RCover *cover, Error ub, Array<Attribute> attributes_to_visit, Attribute last_added,
                            Array<Item> itemset, TrieNode *node, Error lb) {
    //cout << "\t\t lb = " << lb << endl;
    //lb = 0;
    // if (lb > 0) cout << "lb = " << lb << endl;
    if (ub <= lb) {
        // cout << "cc" << endl;
        node->data = query->initData(cover);
        //((QueryData_Best *) node->data)->error = ((QueryData_Best *) node->data)->leafError;
        return node;
    }
    ub = FLT_MAX;
    ncall += 1;
//    cout << "ncall: " << ncall;
//    clock_t tt = clock();
    auto start = high_resolution_clock::now();

    //cout << "tempss: " << (clock() - tt) / (float) CLOCKS_PER_SEC;
    bool verbose = false;
    if (verbose) cout << "ub = " << ub << endl;

    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    Support root_sup = cover->getSupport();
//    cout << "root_sup_class = " << root_sup_clas[0] << ", " << root_sup_clas[1] << endl;

    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for (int m = 0; m < attributes_to_visit.size; ++m) {
        if (last_added == attributes_to_visit[m]) continue;
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
    auto stop = high_resolution_clock::now();
    comptime += duration_cast<milliseconds>(stop - start).count() / 1000.0;
//    cout << " temps comp: " << (clock() - ttt) / (float) CLOCKS_PER_SEC << " ";
//    exit(0);


    Attribute root = -1, left = -1, right = -1;
    Error best_root_error = ub, best_left_error1 = FLT_MAX, best_left_error2 = FLT_MAX, best_right_error1 = FLT_MAX, best_right_error2 = FLT_MAX, best_left_error = FLT_MAX, best_right_error = FLT_MAX;
    Supports best_root_corrects = nullptr, best_left_corrects1 = nullptr, best_left_corrects2 = nullptr, best_right_corrects1 = nullptr, best_right_corrects2 = nullptr, best_left_corrects = nullptr, best_right_corrects = nullptr;
    Supports best_root_falses = nullptr, best_left_falses1 = nullptr, best_left_falses2 = nullptr, best_right_falses1 = nullptr, best_right_falses2 = nullptr, best_left_falses = nullptr, best_right_falses = nullptr;
    Error root_leaf_error = query->computeErrorValues(
            cover).error, best_left_leafError = FLT_MAX, best_right_leafError = FLT_MAX;
    Class best_left_class1 = -1, best_left_class2 = -1, best_right_class1 = -1, best_right_class2 = -1, best_left_class = -1, best_right_class = -1;
    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;
        if (last_added == attr[i]) {
            if (verbose) cout << "pareil que le père...suivant" << endl;
            continue;
        }

        Attribute feat_left = -1, feat_right = -1;
        Error best_feat_left_error = FLT_MAX, best_feat_right_error = FLT_MAX, best_feat_left_error1 = FLT_MAX, best_feat_left_error2 = FLT_MAX, best_feat_right_error1 = FLT_MAX, best_feat_right_error2 = FLT_MAX;
        Error best_feat_left_leafError = FLT_MAX, best_feat_right_leafError = FLT_MAX;
        Class best_feat_left_class1 = -1, best_feat_left_class2 = -1, best_feat_right_class1 = -1, best_feat_right_class2 = -1, best_feat_left_class = -1, best_feat_right_class = -1;
        Supports best_feat_root_corrects = nullptr, best_feat_left_corrects1 = nullptr, best_feat_left_corrects2 = nullptr, best_feat_right_corrects1 = nullptr, best_feat_right_corrects2 = nullptr, best_feat_left_corrects = nullptr, best_feat_right_corrects = nullptr;
        Supports best_feat_root_falses = nullptr, best_feat_left_falses1 = nullptr, best_feat_left_falses2 = nullptr, best_feat_right_falses1 = nullptr, best_feat_right_falses2 = nullptr, best_feat_left_falses = nullptr, best_feat_right_falses = nullptr;
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
        if (igs < query->minsup || ids < query->minsup) {
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }
            // the feature at root cannot be splitted at left. It is then a leaf node
        else if (igs < 2 * query->minsup) {
            ErrorValues ev = query->computeErrorValues(igsc);
            best_feat_left_error = ev.error;
            best_feat_left_class = ev.maxclass;
            best_feat_left_corrects = ev.corrects;
            best_feat_left_falses = ev.falses;
            if (verbose)
                cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = "
                     << best_feat_left_error << " on backtrack" << endl;
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
            if (!floatEqual(ev.error, lb)) {
                Error tmp = best_feat_left_error;
                for (int j = 0; j < attr.size(); ++j) {
                    if (verbose) cout << "left test: " << attr[j] << endl;
                    if (last_added == attr[j] || attr[i] == attr[j]) {
                        if (verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }
                    Supports jdsc = sups[j][j], idjdsc = sups[min(i, j)][max(i, j)], igjdsc = newSupports();
                    subSupports(jdsc, idjdsc, igjdsc);
                    Support jds = sumSupports(jdsc), idjds = sumSupports(idjdsc), igjds = sumSupports(igjdsc);
                    Support igjgs = igs - igjds;

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
                            deleteSupports(tmp_left_corrects2);
                            deleteSupports(tmp_left_falses2);
                            if (verbose)
                                cout << "l'erreur gauche du left montre rien de bon. best root: " << best_root_error
                                     << " best left: " << best_feat_left_error << " Un autre left..." << endl;
                            continue;
                        }

                        Supports igjgsc = newSupports();
                        subSupports(igsc, igjdsc, igjgsc);
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
                            if (verbose)
                                cout << "ce left ci donne une meilleure erreur que les précédents left: "
                                     << best_feat_left_error << endl;
                            best_feat_left_error1 = tmp_left_error1;
                            best_feat_left_error2 = tmp_left_error2;
                            best_feat_left_class1 = tmp_left_class1;
                            best_feat_left_class2 = tmp_left_class2;
                            if (feat_left != -1) {
                                deleteSupports(best_feat_left_corrects1);
                                deleteSupports(best_feat_left_corrects2);
                                deleteSupports(best_feat_left_falses1);
                                deleteSupports(best_feat_left_falses2);
                            }
                            best_feat_left_corrects1 = tmp_left_corrects1;
                            best_feat_left_corrects2 = tmp_left_corrects2;
                            best_feat_left_falses1 = tmp_left_falses1;
                            best_feat_left_falses2 = tmp_left_falses2;
                            feat_left = attr[j];
                            if (floatEqual(best_feat_left_error, lb)) break;
                        } else {
                            deleteSupports(tmp_left_corrects1);
                            deleteSupports(tmp_left_falses1);
                            deleteSupports(tmp_left_corrects2);
                            deleteSupports(tmp_left_falses2);
                            if (verbose)
                                cout << "l'erreur du left = " << tmp_left_error1 + tmp_left_error2
                                     << " n'ameliore pas l'existant. Un autre left..." << endl;
                        }
                        deleteSupports(igjgsc);
                    } else if (verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                    deleteSupports(igjdsc);
                }
                if (floatEqual(best_feat_left_error, tmp) && verbose)
                    cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: "
                         << best_feat_left_error << endl;
            } else {
                if (verbose)
                    cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: "
                         << best_feat_left_error << endl;
            }
        }


        //feature to right
        if (best_feat_left_error < best_root_error) {
            if (verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * query->minsup) {
                ErrorValues ev = query->computeErrorValues(idsc);
                best_feat_right_error = ev.error;
                best_feat_right_class = ev.maxclass;
                best_feat_right_corrects = ev.corrects;
                best_feat_right_falses = ev.falses;
                if (verbose)
                    cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = "
                         << best_feat_right_error << " on backtrack" << endl;
            } else {
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
                if (!floatEqual(ev.error, lb)) {
                    for (int j = 0; j < attr.size(); ++j) {
                        if (verbose) cout << "right test: " << attr[j] << endl;
                        if (last_added == attr[j] || attr[i] == attr[j]) {
                            if (verbose)
                                cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }

                        Supports idjdsc = sups[min(i, j)][max(i, j)], idjgsc = newSupports();
                        subSupports(idsc, idjdsc, idjgsc);
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

                            if (tmp_right_error1 >=
                                min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                deleteSupports(tmp_right_corrects1);
                                deleteSupports(tmp_right_falses1);
                                if (verbose)
                                    cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                continue;
                            }

                            ev = query->computeErrorValues(idjdsc);
                            Error tmp_right_error2 = ev.error;
                            Class tmp_right_class2 = ev.maxclass;
                            Supports tmp_right_corrects2 = ev.corrects;
                            Supports tmp_right_falses2 = ev.falses;
                            if (verbose) cout << "le right a droite produit une erreur de " << tmp_right_error2 << endl;
                            if (tmp_right_error1 + tmp_right_error2 <
                                min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                best_feat_right_error = tmp_right_error1 + tmp_right_error2;
                                plusSupports(tmp_right_corrects1, tmp_right_corrects2, best_feat_right_corrects);
                                plusSupports(tmp_right_falses1, tmp_right_falses2, best_feat_right_falses);
                                if (verbose)
                                    cout << "ce right ci donne une meilleure erreur que les précédents right: "
                                         << best_feat_right_error << endl;
                                best_feat_right_error1 = tmp_right_error1;
                                best_feat_right_error2 = tmp_right_error2;
                                best_feat_right_class1 = tmp_right_class1;
                                best_feat_right_class2 = tmp_right_class2;
                                if (feat_right != -1) {
                                    deleteSupports(best_feat_right_corrects1);
                                    deleteSupports(best_feat_right_corrects2);
                                    deleteSupports(best_feat_right_falses1);
                                    deleteSupports(best_feat_right_falses2);
                                }
                                best_feat_right_corrects1 = tmp_right_corrects1;
                                best_feat_right_corrects2 = tmp_right_corrects2;
                                best_feat_right_falses1 = tmp_right_falses1;
                                best_feat_right_falses2 = tmp_right_falses2;
                                feat_right = attr[j];
                                if (floatEqual(best_feat_right_error, lb)) break;
                            } else {
                                deleteSupports(tmp_right_corrects1);
                                deleteSupports(tmp_right_falses1);
                                deleteSupports(tmp_right_corrects2);
                                deleteSupports(tmp_right_falses2);
                                if (verbose)
                                    cout << "l'erreur du right = " << tmp_right_error1 + tmp_right_error2
                                         << " n'ameliore pas l'existant. Un autre right..." << endl;
                            }
                        } else if (verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        deleteSupports(idjgsc);
                    }
                    if (floatEqual(best_feat_right_error, tmp))
                        if (verbose)
                            cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: "
                                 << best_feat_right_error << endl;
                } else if (verbose)
                    cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: "
                         << best_feat_left_error << endl;
            }

            if (best_feat_left_error + best_feat_right_error < best_root_error) {
//                cout << "o1" << endl;
                best_root_error = best_feat_left_error + best_feat_right_error;
                if (verbose)
                    cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: "
                         << best_root_error << endl;
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
                best_left_corrects = best_feat_left_corrects;
                best_left_falses = best_feat_left_falses;
                best_right_corrects = best_feat_right_corrects;
                best_right_falses = best_feat_right_falses;
                if (feat_left -= -1) {
                    best_left_corrects1 = best_feat_left_corrects1;
                    best_left_falses1 = best_feat_left_falses1;
                    best_left_corrects2 = best_feat_left_corrects2;
                    best_left_falses2 = best_feat_left_falses2;
                }
                if (feat_right -= -1) {
                    best_right_corrects1 = best_feat_right_corrects1;
                    best_right_falses1 = best_feat_right_falses1;
                    best_right_corrects2 = best_feat_right_corrects2;
                    best_right_falses2 = best_feat_right_falses2;
                }
            } else {
//                cout << "o2" << endl;
//                cout << "feat_left = " << feat_left << " and feat_right = " << feat_right << endl;
//                cout << best_left_corrects << endl;
//                if (best_left_corrects) deleteSupports(best_left_corrects);
//                if (best_left_falses) deleteSupports(best_left_falses);
//                if (best_right_corrects) deleteSupports(best_right_corrects);
//                if (best_right_falses) deleteSupports(best_right_falses);
                if (feat_left != -1) {
                    if (best_feat_left_corrects1) deleteSupports(best_feat_left_corrects1);
                    if (best_feat_left_falses1) deleteSupports(best_feat_left_falses1);
                    if (best_feat_left_corrects2) deleteSupports(best_feat_left_corrects2);
                    if (best_feat_left_corrects2) deleteSupports(best_feat_left_falses2);
                }
                if (feat_right != -1) {
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
    if (verbose)
        cout << "le1: " << best_left_error1 << " le2: " << best_left_error2 << " re1: " << best_right_error1 << " re2: "
             << best_right_error2 << endl;
    if (verbose)
        cout << "ble: " << best_left_error << " bre: " << best_right_error << " broe: " << best_root_error << endl;
    if (verbose)
        cout << "lc1: " << best_left_class1 << " lc2: " << best_left_class2 << " rc1: " << best_right_class1 << " rc2: "
             << best_right_class2 << endl;
    if (verbose) cout << "blc: " << best_left_class << " brc: " << best_right_class << endl;
//    cout << "temps find: " << (clock() - tt) / (float) CLOCKS_PER_SEC << " ";

    if (root != -1) {
//            cout << "cc0" << endl;
        //insert root to left
        Array<Item> root_neg;
        root_neg.alloc(itemset.size + 1);
        addItem(itemset, item(root, 0), root_neg);
        TrieNode *root_neg_node = trie->insert(root_neg);
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
        } else {
            ((QueryData_Best *) root_neg_node->data)->test = left;
            ((QueryData_Best *) root_neg_node->data)->leafError = best_left_leafError;
            ((QueryData_Best *) root_neg_node->data)->size = 3;
        }
//            cout << "cc1*" << endl;

        //insert root to right
        Array<Item> root_pos;
        root_pos.alloc(itemset.size + 1);
        addItem(itemset, item(root, 1), root_pos);
        TrieNode *root_pos_node = trie->insert(root_pos);
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
        } else {
            ((QueryData_Best *) root_pos_node->data)->test = right;
            ((QueryData_Best *) root_pos_node->data)->leafError = best_right_leafError;
            ((QueryData_Best *) root_pos_node->data)->size = 3;
        }

//        itemset.free();
//            cout << "cc0*" << endl;

        if (left != -1) {
//                cout << "cc00" << endl;
            //insert left neg
            Array<Item> left_neg;
            left_neg.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 0), left_neg);
            TrieNode *left_neg_node = trie->insert(left_neg);
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
            Array<Item> left_pos;
            left_pos.alloc(root_neg.size + 1);
            addItem(root_neg, item(left, 1), left_pos);
            TrieNode *left_pos_node = trie->insert(left_pos);
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

        if (right != -1) {
//                cout << "cc000" << endl;
            //insert right neg
            Array<Item> right_neg;
            right_neg.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 0), right_neg);
            TrieNode *right_neg_node = trie->insert(right_neg);
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
            Array<Item> right_pos;
            right_pos.alloc(root_pos.size + 1);
            addItem(root_pos, item(right, 1), right_pos);
            TrieNode *right_pos_node = trie->insert(right_pos);
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
        ((QueryData_Best *) node->data)->size =
                ((QueryData_Best *) root_neg_node->data)->size + ((QueryData_Best *) root_pos_node->data)->size + 1;
        ((QueryData_Best *) node->data)->left = (QueryData_Best *) root_neg_node->data;
        ((QueryData_Best *) node->data)->right = (QueryData_Best *) root_pos_node->data;
        ((QueryData_Best *) node->data)->corrects = newSupports();
        plusSupports(((QueryData_Best *) node->data)->left->corrects, ((QueryData_Best *) node->data)->right->corrects,
                     ((QueryData_Best *) node->data)->corrects);
        ((QueryData_Best *) node->data)->falses = newSupports();
        plusSupports(((QueryData_Best *) node->data)->left->falses, ((QueryData_Best *) node->data)->right->falses,
                     ((QueryData_Best *) node->data)->falses);

//            cout << "cc1" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        stop = high_resolution_clock::now();
        spectime += duration_cast<milliseconds>(stop - start).count() / 1000.0;
        return node;
    } else {
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
        stop = high_resolution_clock::now();
        spectime += duration_cast<milliseconds>(stop - start).count() / 1000.0;
        return node;
    }
}



