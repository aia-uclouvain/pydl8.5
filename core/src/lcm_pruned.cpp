#include "lcm_pruned.h"

using namespace std::chrono;


LcmPruned::LcmPruned(RCover *cover, Query *query, bool infoGain, bool infoAsc, bool repeatSort) :
        query(query), cover(cover), infoGain(infoGain), infoAsc(infoAsc), repeatSort(repeatSort) {
}

LcmPruned::~LcmPruned(){}

// the solution already exists for this node
TrieNode *existingsolution(TrieNode *node, Error *nodeError) {
    Logger::showMessageAndReturn("the solution exists and it is worth : ", *nodeError);
    return node;
}

// the node does not fullfil the constraints to be splitted (minsup, depth, etc.)
TrieNode *cannotsplitmore(TrieNode *node, Error ub, Error *nodeError, Error leafError) {
    Logger::showMessageAndReturn("max depth reached. ub = ", ub, " and leaf error = ", leafError);
    // we return the leaf error as node error without checking the upperbound constraint. The parent will do it
    *nodeError = leafError;
    return node;
}

// the node error is equal to the lower bound
TrieNode *reachlowest(TrieNode *node, Error *nodeError, Error leafError) {
    *nodeError = leafError;
    Logger::showMessageAndReturn("lowest error. node error = leaf error = ", *nodeError);
    return node;
}

// the upper bound of the node is lower than the lower bound
TrieNode *infeasiblecase(TrieNode *node, Error *saved_lb, Error ub) {
    Logger::showMessageAndReturn("no solution bcoz ub < lb. lb =", *saved_lb, " and ub = ", ub);
    return node;
}

TrieNode *getSolutionIfExists(TrieNode *node, RCover* cover, Query* query, Error ub, Depth depth){
    Error *nodeError = &(((QDB) node->data)->error);
    // in case the solution exists because the error of a newly created node is set to FLT_MAX
    if (*nodeError < FLT_MAX) {
        return existingsolution(node, nodeError);
    }

    Error *saved_lb = &(((QDB) node->data)->lowerBound);
    // in case the problem is infeasible
    if (ub <= *saved_lb) {
        return infeasiblecase(node, saved_lb, ub);
    }

    Error leafError = ((QDB) node->data)->leafError;
    // we reach the lowest value possible. implicitely, the upper bound constraint is not violated
    if (floatEqual(leafError, *saved_lb)) {
        return reachlowest(node, nodeError, leafError);
    }

    // we cannot split tne node
    if (depth == query->maxdepth || cover->getSupport() < 2 * query->minsup) {
        return cannotsplitmore(node, ub, nodeError, leafError);
    }

    // if time limit is reached we backtrack
    if (query->timeLimitReached) {
        *nodeError = leafError;
        return node;
    }

    return nullptr;
}

// information gain calculation
float LcmPruned::informationGain(Supports notTaken, Supports taken) {
    int sumSupNotTaken = sumSupports(notTaken);
    int sumSupTaken = sumSupports(taken);
    int actualDBSize = sumSupNotTaken + sumSupTaken;

    float condEntropy = 0, baseEntropy = 0;
    float priorProbNotTaken = (actualDBSize != 0) ? (float) sumSupNotTaken / actualDBSize : 0;
    float priorProbTaken = (actualDBSize != 0) ? (float) sumSupTaken / actualDBSize : 0;
    float e0 = 0, e1 = 0;

    for (int j = 0; j < cover->dm->getNClasses(); ++j) {
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


Array<Attribute> LcmPruned::getSuccessors(Array<Attribute> last_candidates, Attribute last_added) {

    std::multimap<float, Attribute> gain;
    Array<Attribute> next_candidates(last_candidates.size, 0);

    // the current node does not fullfill the frequency criterion. In correct situation, this case won't happen
    if (cover->getSupport() < 2 * query->minsup)
        return next_candidates;

    int current_sup = cover->getSupport();
    Supports current_sup_class = cover->getSupportPerClass();

    // access each candidate
    for (auto& candidate : last_candidates) {

        // this attribute is already in the current itemset
        if (last_added == candidate) continue;

        // compute the support of each candidate
        int sup_left = cover->temporaryIntersectSup(candidate, false);
        int sup_right = current_sup - sup_left; //no need to intersect with negative item to compute its support

        // add frequent attributes but if heuristic is used to sort them, compute its value and sort later
        if (sup_left >= query->minsup && sup_right >= query->minsup) {
            //continuous dataset. Not supported yet
//            if (query->continuous) {}
//            else {
                if (infoGain) {
                    // compute the support per class in each split of the attribute to compute its IG value
                    Supports sup_class_left = cover->temporaryIntersect(candidate, false).first;
                    Supports sup_class_right = newSupports();
                    subSupports(current_sup_class, sup_class_left, sup_class_right);
                    gain.insert(std::pair<float, Attribute>(informationGain(sup_class_left, sup_class_right),
                                                            candidate));
                    deleteSupports(sup_class_left);
                    deleteSupports(sup_class_right);
                } else next_candidates.push_back(candidate);
//            }
        }
    }

    // if heuristic is used, add the next candidates given the heuristic order
    if (infoGain) {
        if (infoAsc) for (auto & it : gain) next_candidates.push_back(it.second); //items with low IG first
        else for (auto it = gain.rbegin(); it != gain.rend(); ++it) next_candidates.push_back(it->second); //items with high IG first
    }
    // disable the heuristic variable if the sort must be performed once
    if (!repeatSort) infoGain = false;

    return next_candidates;
}

// find the successors of a node when it has been already visited in the past
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

// compute the similarity lower bound based on the best ever seen node or the node with the highest coversize
Error LcmPruned::computeSimilarityLowerBound(bitset<M> *b1_cover, bitset<M> *b2_cover, Error b1_error, Error b2_error) {
//    return 0;
    if (is_python_error) return 0;
    Error bound = 0;
    bitset<M>*covers[] = {b1_cover, b2_cover};
    Error errors[] = {b1_error, b2_error};
    for (int i : {0, 1}) {
        bitset<M>* cov = covers[i];
        Error err = errors[i];
        if (cov) {
            SupportClass sumdif = cover->countDif(cov);
            if (err - sumdif > bound) bound = err - sumdif;
        }
    }
    return (bound > 0) ? bound : 0;
}

// store the node with lowest error as well as the one with the largest cover in order to find a similarity lower bound
void LcmPruned::addInfoForLowerBound(QueryData *node_data, bitset<M> *&b1_cover, bitset<M> *&b2_cover,
                                    Error &b1_error, Error &b2_error, Support &highest_coversize) {
//    if (((QDB) node_data)->error < FLT_MAX) {
    Error err = (((QDB) node_data)->error < FLT_MAX) ? ((QDB) node_data)->error : ((QDB) node_data)->lowerBound;
    Support sup = cover->getSupport();

    if (err < FLT_MAX && err > b1_error) {
        delete[] b1_cover;
        b1_cover = cover->getTopBitsetArray();
        b1_error = err;
    }

    if (sup > highest_coversize) {
        delete[] b2_cover;
        b2_cover = cover->getTopBitsetArray();
        b2_error = err;
        highest_coversize = sup;
    }
//    }
}

/** recurse - this method finds the best tree given an itemset and its cover and update
 * the information of the node representing the itemset. Each itemset is represented by a node and info about the
 * tree structure is wrapped into a variable data in the node object. Each itemset (the node) is inserted into the
 * trie (if it had not been inserted) before the call to the current function. When it has not been evaluated, the
 * data variable is set to null otherwise it contains info wrapped into QDB object
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
                             Depth depth,
                             float ub,
                             float computed_lb) {

    // check if we ran out of time
    if (query->timeLimit > 0) {
        float runtime = duration_cast<milliseconds>(high_resolution_clock::now() - query->startTime).count() / 1000.0;
        if (runtime >= query->timeLimit)
            query->timeLimitReached = true;
    }

    // the node data already exists because it is not null like how it is when it is just created
    if (node->data) {
        Logger::showMessageAndReturn("the node exists");
        TrieNode* result = getSolutionIfExists(node, cover, query, ub, depth);
        if (result) return result;
    }

    // in case the solution cannot be derived without computation and remaining depth is 2, we use a specific algorithm
    if (query->maxdepth - depth == 2 && cover->getSupport() >= 2 * query->minsup && no_python_error) {
        return computeDepthTwo(cover, ub, next_candidates, last_added, itemset, node, query, computed_lb, query->trie);
    }

    /* there are two cases in which the execution attempt here
     1- when the node data did not exist
     2- when the node data exists without solution and its upper bound is higher than its lower bound*/


    /* at this stage, we will probably make a search through successors, so we create empty array of them.
     It will be replaced by the good one after calling getSuccessors function*/
    Array<Attribute> next_attributes;

    /* case 1 : the node data did not exist
     no need to insert the node into the trie. It has just been created and inserted into the trie
     before the call to this function. we will just the data object (QDB) and its information*/
    if (!node->data) {
        Logger::showMessageAndReturn("New node");
        latticesize++;

        // Create data object and initialize its variables, then get them for the search
        node->data = query->initData(cover);
        Logger::showMessageAndReturn("after init of the new node. ub = ", ub, " and leaf error = ", ((QDB) node->data)->leafError);
        TrieNode* result = getSolutionIfExists(node, cover, query, ub, depth);
        if (result) return result;

        // if we can't get solution without computation, we compute the next candidates to perform the search
        next_attributes = getSuccessors(next_candidates, last_added);
    }
    //case 2 : the node data exists without solution but ub > last ub which is now lb
    else {
        Error leafError = ((QDB) node->data)->leafError;
        Error *nodeError = &(((QDB) node->data)->error);
        Logger::showMessageAndReturn("existing node without solution and higher bound. leaf error = ", leafError, " new ub = ", ub);

        if (query->timeLimitReached) {
            *nodeError = leafError;
            return node;
        }

        // if we can't get solution without computation, we compute the next candidates to perform the search
        next_attributes = getSuccessors(next_candidates, last_added);
        // next_attributes = getExistingSuccessors(node);
        // next_attributes = getSuccessors(next_candidates, cover, last_added);
    }

    // as the best tree cannot be deducted without computation, we compute the search

    Error *lb = &(((QDB) node->data)->lowerBound);
    Error leafError = ((QDB) node->data)->leafError;
    Error *nodeError = &(((QDB) node->data)->error);

    // case in which there is no candidate
    if (next_attributes.size == 0) {
        Logger::showMessageAndReturn("No candidates. nodeError is set to leafError");
        *nodeError = leafError;
        Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", *nodeError);
        Logger::showMessageAndReturn("we backtrack");
        next_attributes.free();
        return node;
    }

    // parameters for similarity lower bound
    bool first_item = false, second_item = true;
    bitset<M> *b1_cover = nullptr, *b2_cover = nullptr;
    // Supports b1_sc = nullptr, b2_sc = nullptr;
    Error b1_error = 0, b2_error = 0;
    Support highest_coversize = 0;
    // in case solution, is not found, this value is the minimum of the minimum error
    // for each attribute. It can be used as a lower bound
    Error minlb = FLT_MAX;

    //bount for the first child (item)
    Error child_ub = ub;

    // we evaluate the split on each candidate attribute
    for(auto& next : next_attributes) {
        Logger::showMessageAndReturn("\n\nWe are evaluating the attribute : ", next);

        Array<Item> itemsets[2];
        TrieNode *nodes[2];
        Error first_lb = 0, second_lb = 0;

        /* the lower bound is computed for both items. they are used as heuristic to decide
         the first item to branch on. We branch on item with higher lower bound to have chance
         to get a higher error to violate the ub constraint and prune the second branch
         this computation is costly so, in some case can add some overhead. If you don't
         want to use it, please comment the next block. 0/1 order is used in this case.*/

        //=========================== BEGIN BLOCK ==========================//
        cover->intersect(next, false);
        first_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
        cover->backtrack();

        cover->intersect(next);
        second_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
        cover->backtrack();
        //=========================== END BLOCK ==========================//


        first_item = second_lb > first_lb;
        second_item = !first_item;

        // perform search on the first item
        cover->intersect(next, first_item);
        itemsets[first_item] = addItem(itemset, item(next, first_item));
        nodes[first_item] = query->trie->insert(itemsets[first_item]);
        // if lower bound was not computed
        if (floatEqual(first_lb, -1)) first_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
        // the best lower bound between the computed and the saved is used
        first_lb = (nodes[first_item]->data) ? max(((QDB) nodes[first_item]->data)->lowerBound, first_lb) : first_lb;
        // perform the search for the first item
        nodes[first_item] = recurse(itemsets[first_item], next, nodes[first_item], next_attributes,  depth + 1, child_ub - second_lb, first_lb);

        // check if the found information is relevant to compute the next similarity bounds
        addInfoForLowerBound(nodes[first_item]->data, b1_cover, b2_cover, b1_error, b2_error, highest_coversize);
        //cout << "after good bound 1" << " sc[0] = " << b1_sc[0] << " sc[1] = " << b1_sc[1] << " err = " << ((QDB)nodes[first_item]->data)->error << endl;
        Error firstError = ((QDB) nodes[first_item]->data)->error;
        itemsets[first_item].free();
        cover->backtrack();

        if (query->canimprove(nodes[first_item]->data, child_ub)) {
            // perform search on the second item
            cover->intersect(next, second_item);
            itemsets[second_item] = addItem(itemset, item(next, second_item));
            nodes[second_item] = query->trie->insert(itemsets[second_item]);
            if (floatEqual(second_lb, -1)) second_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
            // the best lower bound between the computed and the saved is used
            second_lb = (nodes[second_item]->data) ? max(((QDB) nodes[second_item]->data)->lowerBound, second_lb) : second_lb;
            // bound for the second child (item)
            Error remainUb = child_ub - firstError;
            // perform the search for the second item
            nodes[second_item] = recurse(itemsets[second_item], next, nodes[second_item], next_attributes, depth + 1, remainUb, second_lb);

            // check if the found information is relevant to compute the next similarity bounds
            addInfoForLowerBound(nodes[second_item]->data, b1_cover, b2_cover, b1_error, b2_error, highest_coversize);
            Error secondError = ((QDB) nodes[second_item]->data)->error;
            itemsets[second_item].free();
            cover->backtrack();

            Error feature_error = firstError + secondError;
            bool hasUpdated = query->updateData(node->data, child_ub, next, nodes[0]->data, nodes[1]->data);
            if (hasUpdated) {
                child_ub = feature_error;
                Logger::showMessageAndReturn("-\nafter this attribute, node error=", *nodeError, " and ub=", child_ub);
            }
            // in case we get the real error, we update the minimum possible error
            else minlb = min(minlb, feature_error);

            if (query->canSkip(node->data) || query->timeLimitReached) {//lowerBound reached
                Logger::showMessageAndReturn("We get the best solution or no more time. So, we break the remaining attributes");
                break; //prune remaining attributes not browsed yet
            }
        } else { //we do not attempt the second child, so we use its lower bound

            // if the first error is unknown, we use its lower bound
            if (floatEqual(firstError, FLT_MAX)) minlb = min(minlb, first_lb + second_lb);
            // otherwise, we use it
            else minlb = min(minlb, firstError + second_lb);
        }

        if (query->stopAfterError) {
            if (depth == 0 && ub < FLT_MAX) {
                if (*nodeError < ub)
                    break;
            }
        }
    }
    delete[] b1_cover;
    delete[] b2_cover;

    // we do not get solution and new lower bound is better than the old
    if (floatEqual(*nodeError, FLT_MAX) && max(ub, minlb) > *lb) {
        *lb = max(ub, minlb);
    }

    Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", *nodeError);

    next_attributes.free();
//        itemset.free();
    return node;


}


void LcmPruned::run() {
    query->setStartTime();
    // set the correct maxerror if needed
    float maxError = NO_ERR;
    if (query->maxError > 0) maxError = query->maxError;

    // Create empty list for candidate attributes
    Array<Attribute> attributes_to_visit(nattributes, 0);

    // Update the candidate list based on frequency criterion
    if (query->minsup == 1) { // do not check frequency if minsup = 1
        for (int attr = 0; attr < nattributes; ++attr) attributes_to_visit.push_back(attr);
    }
    else { // make sure each candidate attribute can be split into two nodes fulfilling the frequency criterion
        for (int attr = 0; attr < nattributes; ++attr) {
            if (cover->temporaryIntersectSup(attr, false) >= query->minsup && cover->temporaryIntersectSup(attr) >= query->minsup)
                attributes_to_visit.push_back(attr);
        }
    }

    //create an empty array of items representing an emptyset and insert it
    Array<Item> itemset;
    itemset.size = 0;
    itemset.elts = nullptr;

    // insert the emptyset node
    TrieNode *node = query->trie->insert(itemset);

    // call the recursive function to start the search
    query->realroot = recurse(itemset, NO_ATTRIBUTE, node, attributes_to_visit, 0, maxError);

    // never forget to return back what is not yours. Think to others who need it ;-)
    itemset.free();
    attributes_to_visit.free();

    /*cout << "ncall: " << ncall << endl;
    cout << "comptime: " << comptime << endl;
    cout << "searchtime: " << spectime << endl;
    cout << "totaltime: " << comptime + spectime << endl;
    ncall = 0; comptime = 0; spectime = 0;*/
}