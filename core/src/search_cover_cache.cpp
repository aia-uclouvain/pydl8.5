#include "search_cover_cache.h"

using namespace std::chrono;


Search_cover_cache::Search_cover_cache(NodeDataManager *nodeDataManager, bool infoGain, bool infoAsc, bool repeatSort,
                                       Support minsup,
                                       Depth maxdepth,
                                       int timeLimit,
                                       Cache *cache,
                                       float maxError,
                                       bool specialAlgo,
                                       bool stopAfterError,
                                       bool similarlb,
                                       bool dynamic_branching,
                                       bool similar_for_branching,
                                       bool from_cpp) :
        Search_base(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, cache, maxError, specialAlgo, stopAfterError, from_cpp), similarlb(similarlb), dynamic_branching(dynamic_branching), similar_for_branching(similar_for_branching) {}

Search_cover_cache::~Search_cover_cache() {}

// the solution already exists for this node
Node *existingsolution_(Node *node, Error *nodeError) {
    Logger::showMessageAndReturn("the solution exists and it is worth : ", *nodeError);
    return node;
}

// the node does not fullfil the constraints to be splitted (minsup, depth, etc.)
Node *cannotsplitmore_(Node *node, Error ub, Error *nodeError, Error leafError) {
    Logger::showMessageAndReturn("max depth reached. ub = ", ub, " and leaf error = ", leafError);
    // we return the leaf error as node error without checking the upperbound constraint. The parent will do it
    *nodeError = leafError;
    return node;
}

// the node error is equal to the lower bound
Node *reachlowest_(Node *node, Error *nodeError, Error leafError) {
    *nodeError = leafError;
    Logger::showMessageAndReturn("lowest error. node error = leaf error = ", *nodeError);
    return node;
}

// the upper bound of the node is lower than the lower bound
Node *infeasiblecase_(Node *node, Error *saved_lb, Error ub) {
    Logger::showMessageAndReturn("no solution bcoz ub < lb. lb =", *saved_lb, " and ub = ", ub);
    return node;
}

Node * Search_cover_cache::getSolutionIfExists(Node *node, Error ub, Depth depth){

    Error *nodeError = &(((CoverNodeData*) node->data)->error);
    if (*nodeError < FLT_MAX) return existingsolution_(node, nodeError); // solution exists (new node error is FLT_MAX)

    Error *saved_lb = &(((CoverNodeData*) node->data)->lowerBound);
    // in case the problem is infeasible
    if (ub <= *saved_lb or ub <= 0) {
        return infeasiblecase_(node, saved_lb, ub);
    }

    Error leafError = ((CoverNodeData*) node->data)->leafError;
    // we reach the lowest value possible. implicitely, the upper bound constraint is not violated
    if (floatEqual(leafError, *saved_lb)) {
        return reachlowest_(node, nodeError, leafError);
    }

    // we cannot split the node
    if (depth == maxdepth || nodeDataManager->cover->getSupport() < 2 * minsup) {
        return cannotsplitmore_(node, ub, nodeError, leafError);
    }

    // if time limit is reached we backtrack
    if (timeLimitReached) {
        *nodeError = leafError;
        return node;
    }

    return nullptr;
}

Node * Search_cover_cache::inferSolutionFromLB(Node *node, Error ub){

    Error *nodeError = &(((CoverNodeData*) node->data)->error);
    Error *saved_lb = &(((CoverNodeData*) node->data)->lowerBound);
    if (ub <= *saved_lb) return infeasiblecase_(node, saved_lb, ub); // infeasible case

    Error leafError = ((CoverNodeData*) node->data)->leafError;
    if (floatEqual(leafError, *saved_lb)) return reachlowest_(node, nodeError, leafError); // lowest possible value reached

    if (timeLimitReached) { *nodeError = leafError; return node; } // time limit reached

    return nullptr;
}

// information gain calculation
float Search_cover_cache::informationGain(ErrorVals notTaken, ErrorVals taken) {
    int sumSupNotTaken = sumErrorVals(notTaken);
    int sumSupTaken = sumErrorVals(taken);
    int actualDBSize = sumSupNotTaken + sumSupTaken;

    float condEntropy = 0, baseEntropy = 0;
    float priorProbNotTaken = (actualDBSize != 0) ? (float) sumSupNotTaken / actualDBSize : 0;
    float priorProbTaken = (actualDBSize != 0) ? (float) sumSupTaken / actualDBSize : 0;
    float e0 = 0, e1 = 0;

    for (int j = 0; j < nodeDataManager->cover->dm->getNClasses(); ++j) {
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


Attributes Search_cover_cache::getSuccessors(Attributes &last_candidates, Attribute last_added) {

    std::multimap<float, Attribute> gain;
    Attributes next_candidates;
    next_candidates.reserve(last_candidates.size() - 1);

    // the current node does not fullfill the frequency criterion. In correct situation, this case won't happen
    if (nodeDataManager->cover->getSupport() < 2 * minsup) return next_candidates;

    int current_sup = nodeDataManager->cover->getSupport();
    ErrorVals current_sup_class = nodeDataManager->cover->getErrorValPerClass();

    // access each candidate
    for (const auto candidate : last_candidates) {

        // this attribute is already in the current itemset
        if (last_added == candidate) continue;

        // compute the support of each candidate
        int sup_left = nodeDataManager->cover->temporaryIntersectSup(candidate, false);
        int sup_right = current_sup - sup_left; //no need to intersect with negative item to compute its support

        // add frequent attributes but if heuristic is used to sort them, compute its value and sort later
        if (sup_left >= minsup && sup_right >= minsup) {
            if (infoGain) {
                // compute the support per class in each split of the attribute to compute its IG value
                ErrorVals sup_class_left = nodeDataManager->cover->temporaryIntersect(candidate, false).first;
                ErrorVals sup_class_right = newErrorVals();
                subErrorVals(current_sup_class, sup_class_left, sup_class_right);
                gain.insert(std::pair<float, Attribute>(informationGain(sup_class_left, sup_class_right), candidate));
                deleteErrorVals(sup_class_left);
                deleteErrorVals(sup_class_right);
            } else next_candidates.push_back(candidate);
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

// compute the similarity lower bound based on the best ever seen node or the node with the highest coversize
Error Search_cover_cache::computeSimilarityLB(SimilarValss &similar_db1, SimilarValss &similar_db2) {
    //return 0;
    if (is_python_error) return 0;
    Error bound = 0;
    bitset<M>* covers[] = {similar_db1.s_cover, similar_db2.s_cover};
    int* validWords[] = {similar_db1.s_validWords, similar_db2.s_validWords};
    int nvalidWords[] = {similar_db1.s_n_validWords, similar_db2.s_n_validWords};
    Error errors[] = {similar_db1.s_error, similar_db2.s_error};
    for (int i : {0, 1}) {
        bitset<M>* cov = covers[i];
        Error err = errors[i];
        int* valid = validWords[i];
        int nvalid = nvalidWords[i];
        if (cov != nullptr) {
            ErrorVal diff_err_val = nodeDataManager->cover->getDiffErrorVal(cov, valid, nvalid);
            bound = max(bound, err - diff_err_val);
        }
    }
    return max(0.f, bound);// (bound > 0) ? bound : 0;
}


void Search_cover_cache::updateSimilarLBInfo2(NodeData *node_data, SimilarValss &similar_db1, SimilarValss &similar_db2) {

    Error err = (((CoverNodeData*) node_data)->error < FLT_MAX) ? ((CoverNodeData*) node_data)->error : ((CoverNodeData*) node_data)->lowerBound;
    Support sup = nodeDataManager->cover->getSupport();

    if (floatEqual(err, 0)) return;

    if (similar_db1.s_cover == nullptr) { // the first db is saved
        similar_db1.s_cover = nodeDataManager->cover->getTopCover();
        similar_db1.s_error = err;
        similar_db1.s_coversize = sup;
        similar_db1.s_n_validWords = nodeDataManager->cover->limit.top();
        similar_db1.s_validWords = new int[similar_db1.s_n_validWords];
        for (int i = 0; i < similar_db1.s_n_validWords; ++i) {
            similar_db1.s_validWords[i] = nodeDataManager->cover->validWords[i];
        }
    } else if (similar_db2.s_cover == nullptr) { // the second db is saved
        similar_db2.s_cover = nodeDataManager->cover->getTopCover();
        similar_db2.s_error = err;
        similar_db2.s_coversize = sup;
        similar_db2.s_n_validWords = nodeDataManager->cover->limit.top();
        similar_db2.s_validWords = new int[similar_db2.s_n_validWords];
        for (int i = 0; i < similar_db2.s_n_validWords; ++i) {
            similar_db2.s_validWords[i] = nodeDataManager->cover->validWords[i];
        }
    } else { // then, the new entering db is replaced by the closest among the two saved
        ErrorVal err_din = nodeDataManager->cover->getDiffErrorVal(similar_db1.s_cover, similar_db1.s_validWords, similar_db1.s_n_validWords, true);
        ErrorVal err_dout = nodeDataManager->cover->getDiffErrorVal(similar_db1.s_cover, similar_db1.s_validWords, similar_db1.s_n_validWords, false);
        ErrorVal cov_din = nodeDataManager->cover->getDiffErrorVal(similar_db2.s_cover, similar_db2.s_validWords, similar_db2.s_n_validWords, true);
        ErrorVal cov_dout = nodeDataManager->cover->getDiffErrorVal(similar_db2.s_cover, similar_db2.s_validWords, similar_db2.s_n_validWords, false);
        if (err_din + err_dout < cov_din + cov_dout){
            delete[] similar_db1.s_cover;
            similar_db1.s_cover = nodeDataManager->cover->getTopCover();
            similar_db1.s_error = err;
            similar_db1.s_coversize = sup;
            delete[] similar_db1.s_validWords;
            similar_db1.s_n_validWords = nodeDataManager->cover->limit.top();
            similar_db1.s_validWords = new int[similar_db1.s_n_validWords];
            for (int i = 0; i < similar_db1.s_n_validWords; ++i) {
                similar_db1.s_validWords[i] = nodeDataManager->cover->validWords[i];
            }
        }
        else {
            delete[] similar_db2.s_cover;
            similar_db2.s_cover = nodeDataManager->cover->getTopCover();
            similar_db2.s_error = err;
            similar_db2.s_coversize = sup;
            delete[] similar_db2.s_validWords;
            similar_db2.s_n_validWords = nodeDataManager->cover->limit.top();
            similar_db2.s_validWords = new int[similar_db2.s_n_validWords];
            for (int i = 0; i < similar_db2.s_n_validWords; ++i) {
                similar_db2.s_validWords[i] = nodeDataManager->cover->validWords[i];
            }
        }
    }

}


// store the node with the highest error as well as the one with the largest cover in order to find a similarity lower bound
void Search_cover_cache::updateSimilarLBInfo1(NodeData *node_data, SimilarValss &highest_error_db, SimilarValss &highest_coversize_db) {

    Error err = (((CoverNodeData*) node_data)->error < FLT_MAX) ? ((CoverNodeData*) node_data)->error : ((CoverNodeData*) node_data)->lowerBound;
    Support sup = nodeDataManager->cover->getSupport();

    if (floatEqual(err, 0)) return;

    if (err < FLT_MAX and err > highest_error_db.s_error) {
        delete[] highest_error_db.s_cover;
        highest_error_db.s_cover = nodeDataManager->cover->getTopCover();
        highest_error_db.s_error = err;
        highest_error_db.s_coversize = sup;
        delete[] highest_error_db.s_validWords;
        highest_error_db.s_n_validWords = nodeDataManager->cover->limit.top();
        highest_error_db.s_validWords = new int[highest_error_db.s_n_validWords];
        for (int i = 0; i < highest_error_db.s_n_validWords; ++i) {
            highest_error_db.s_validWords[i] = nodeDataManager->cover->validWords[i];
        }
        return;
    }

    if (err < FLT_MAX and sup > highest_coversize_db.s_coversize) {
        delete[] highest_coversize_db.s_cover;
        highest_coversize_db.s_cover = nodeDataManager->cover->getTopCover();
        highest_coversize_db.s_error = err;
        highest_coversize_db.s_coversize = sup;
        delete[] highest_coversize_db.s_validWords;
        highest_coversize_db.s_n_validWords = nodeDataManager->cover->limit.top();
        highest_coversize_db.s_validWords = new int[highest_coversize_db.s_n_validWords];
        for (int i = 0; i < highest_coversize_db.s_n_validWords; ++i) {
            highest_coversize_db.s_validWords[i] = nodeDataManager->cover->validWords[i];
        }
    }
}

/** recurse - this method finds the best tree given an itemset and its cover and update
 * the information of the node representing the itemset. Each itemset is represented by a node and info about the
 * tree structure is wrapped into a variable data in the node object. Each itemset (the node) is inserted into the
 * trie (if it had not been inserted) before the call to the current function. When it has not been evaluated, the
 * data variable is set to null otherwise it contains info wrapped into CoverNodeData* object
 *
 * @param itemset - the itemset for which we are looking for the best tree
 * @param last_added - the last added attribute
 * @param node - the node representing the itemset
 * @param next_candidates - next attributes to visit
 * @param depth - the current depth in the search tree
 * @param ub - the upper bound of the search. It cannot be reached
 * @return the same node as get in parameter with added information about the best tree
 */
pair<Node*,HasInter> Search_cover_cache::recurse(Itemset &itemset,
                                                 Item last_added_item,
                                                 Node *node,
                                                 bool node_is_new,
                                                 Attributes &next_candidates,
                                                 Depth depth,
                                                 Error ub,
                                                 SimilarValss &sim_db1,
                                                 SimilarValss &sim_db2) {

    // check if we ran out of time
    if (timeLimit > 0 and duration<float>(high_resolution_clock::now() - GlobalParams::getInstance()->startTime).count() >= (float)timeLimit) timeLimitReached = true;

    Node* result = getSolutionIfExists(node, ub, depth);
    if (result) { // the solution can be inferred without computation
        return {result, false}; // the second value is to state whether an intersection has been performed or not
    }
    Logger::showMessageAndReturn("Node solution cannot be found without calculation");

    // in case of root (empty item), there is no last added attribute
    Attribute last_added_attr = (last_added_item == NO_ITEM) ? NO_ATTRIBUTE : item_attribute(last_added_item);

    // in case the solution cannot be derived without computation and remaining depth is 2, we use a specific algorithm
    if (specialAlgo and maxdepth - depth == 2 and nodeDataManager->cover->getSupport() >= 2 * minsup and no_python_error) {
//        if (itemset.size() == 4 and itemset.at(0) == 4 and itemset.at(1) == 29 and itemset.at(2) == 35 and itemset.at(3) == 47) {
//            GlobalParams::getInstance()->verbose = true;
//        }
        computeDepthTwo(nodeDataManager->cover, ub, next_candidates, last_added_attr, itemset, node, nodeDataManager, ((CoverNodeData*) node->data)->lowerBound, cache, this, true);
//        if (itemset.size() == 4 and itemset.at(0) == 4 and itemset.at(1) == 29 and itemset.at(2) == 35 and itemset.at(3) == 47) {
//            GlobalParams::getInstance()->verbose = false;
//        }
        return {node, true};
    }

    /* the node solution cannot be computed without calculation. at this stage, we will make a search through successors*/
    Error leafError = ((CoverNodeData*) node->data)->leafError;
    Error *nodeError = &(((CoverNodeData*) node->data)->error);
    Logger::showMessageAndReturn("leaf error = ", leafError, " ub = ", ub);

    if (timeLimitReached) { *nodeError = leafError; return {node, true}; }

    // if we can't get solution without computation, we compute the next candidates to perform the search
    Attributes next_attributes = getSuccessors(next_candidates, last_added_attr);

    // case in which there is no candidate
    if (next_attributes.empty()) {
        *nodeError = leafError;
        Logger::showMessageAndReturn("No candidates. nodeError is set to leafError\n", "depth = ", depth, " and init ub = ", ub, " and error after search = ", *nodeError, "\nwe backtrack");
        return {node, true};
    }

    // parameters for similarity lower bound
    SimilarValss similar_db1, similar_db2;
    /* in case solution is not found, this value is the minimum of lower bounds
     * error of each attribute(sum per item). It can be used as a lower bound for the current node*/
    Error minlb = FLT_MAX, child_ub = ub; // upper bound for the first child (item)
    bool first_item, second_item; // the best feature for the current node

    // we evaluate the split on each candidate attribute
    for(const auto &attr : next_attributes) {
        Logger::showMessageAndReturn("\n\nWe are evaluating the attribute : ", attr);

        Itemset itemsets[2];
        Node *child_nodes[2];
        Error neg_lb = 0, pos_lb = 0;
        Error first_lb, second_lb;

        /* the lower bound is computed for both items. they are used as heuristic to decide
         the first item to branch on. We branch on item with higher lower bound to have chance
         to get a higher error to violate the ub constraint and prune the second branch
         this computation can be costly so, in some case can add some overhead. If you don't
         want to use it, set dynamic_branching to false. 0/1 order is used in this case.*/
        if (dynamic_branching) {

            nodeDataManager->cover->intersect(attr, NEG_ITEM);
            Node* tmp_node = cache->get(nodeDataManager, itemset.size() + 1);
            if (tmp_node != nullptr and tmp_node->data != nullptr) {
                neg_lb = tmp_node->data->error < FLT_MAX ? tmp_node->data->error : tmp_node->data->lowerBound;
            }
            if (similarlb and similar_for_branching) neg_lb = max(neg_lb, computeSimilarityLB(similar_db1, similar_db2));
            nodeDataManager->cover->backtrack();

            nodeDataManager->cover->intersect(attr, POS_ITEM);
            tmp_node = cache->get(nodeDataManager, itemset.size() + 1);
            if (tmp_node != nullptr and tmp_node->data != nullptr) {
                pos_lb = tmp_node->data->error < FLT_MAX ? tmp_node->data->error : tmp_node->data->lowerBound;
            }
            if (similarlb and similar_for_branching) pos_lb = max(pos_lb, computeSimilarityLB(similar_db1, similar_db2));
            nodeDataManager->cover->backtrack();
        }

        // the first item is the one with the highest lower bound
        first_item = pos_lb > neg_lb;
        second_item = not first_item;
        first_lb = (first_item) ? pos_lb : neg_lb;
        second_lb = (not floatEqual(first_lb, pos_lb)) ? pos_lb : neg_lb;

        // perform search on the first item
        itemsets[first_item] = addItem(itemset, item(attr, first_item));
        nodeDataManager->cover->intersect(attr, first_item);
        pair<Node*, bool> node_state = cache->insert(nodeDataManager, itemsets[first_item].size());
//        pair<Node*, bool> node_state = cache->insert(nodeDataManager, itemsets[first_item].size(), itemsets[first_item]);
        child_nodes[first_item] = node_state.get_node;
        ((CoverNodeData*)node->data)->curr_left = (HashCoverNode*)node_state.get_node;
        if (node_state.is_new){
            child_nodes[first_item]->data = nodeDataManager->initData();
            Logger::showMessageAndReturn("Newly created node node. leaf error = ", ((CoverNodeData*) child_nodes[first_item]->data)->leafError);
        } else Logger::showMessageAndReturn("The node already exists");
        child_nodes[first_item]->data->lowerBound = first_lb; // the best lb between the computed and the saved ones is selected
        pair<Node*, HasInter> node_inter = recurse(itemsets[first_item], item(attr, first_item), child_nodes[first_item], node_state.is_new, next_attributes,  depth + 1, child_ub - second_lb, similar_db1, similar_db2); // perform the search for the first item
        child_nodes[first_item] = node_inter.get_node;
        if (similarlb) updateSimilarLBInfo2(child_nodes[first_item]->data, similar_db1, similar_db2);
        nodeDataManager->cover->backtrack();
        Error firstError = child_nodes[first_item]->data->error;
        Itemset().swap(itemsets[first_item]); // fre the vector memory representing the first itemset


        if (nodeDataManager->canimprove(child_nodes[first_item]->data, child_ub - second_lb)) { // perform search on the second item
            itemsets[second_item] = addItem(itemset, item(attr, second_item));
            nodeDataManager->cover->intersect(attr, second_item);
            node_state = cache->insert(nodeDataManager, itemsets[second_item].size());
//            node_state = cache->insert(nodeDataManager, itemsets[second_item].size(), itemsets[second_item]);
            child_nodes[second_item] = node_state.get_node;
            ((CoverNodeData*)node->data)->curr_right = (HashCoverNode*)node_state.get_node;
            if (node_state.is_new){
                child_nodes[second_item]->data = nodeDataManager->initData();
                Logger::showMessageAndReturn("Newly created node node. leaf error = ", ((CoverNodeData*) child_nodes[second_item]->data)->leafError);
            } else Logger::showMessageAndReturn("The node already exists");
            child_nodes[second_item]->data->lowerBound = second_lb; // the best lb between the computed and the saved ones is selected
            Error remainUb = child_ub - firstError; // bound for the second child (item)
            node_inter = recurse(itemsets[second_item], item(attr, second_item), child_nodes[second_item], node_state.is_new, next_attributes,  depth + 1, remainUb, similar_db1, similar_db2); // perform the search for the second item
            child_nodes[second_item] = node_inter.get_node;
            if (similarlb) updateSimilarLBInfo2(child_nodes[second_item]->data, similar_db1, similar_db2);
            nodeDataManager->cover->backtrack();
            Error secondError = child_nodes[second_item]->data->error;
            Itemset().swap(itemsets[second_item]); // fre the vector memory representing the second itemset

            Error feature_error = firstError + secondError;
            bool hasUpdated = nodeDataManager->updateData(node, child_ub, attr, child_nodes[NEG_ITEM], child_nodes[POS_ITEM]);
            if (hasUpdated) {
                child_ub = feature_error;
                Logger::showMessageAndReturn("-after this attribute ", attr, ", node error=", *nodeError, " and ub=", child_ub);
            }
            else { // in case we get the real error, we update the minimum possible error
                minlb = min(minlb, feature_error);
            }

            if (nodeDataManager->canSkip(node->data) or timeLimitReached) { //lowerBound or time limit reached
                Logger::showMessageAndReturn("We get the best solution. So, we break the remaining attributes");
                break; //prune remaining attributes not browsed yet
            }
        }
        else { //we do not attempt the second child, so we use its lower bound
            // if the first error is unknown, we use its lower bound. otherwise, we use it
            if (floatEqual(firstError, FLT_MAX)) minlb = min(minlb, child_nodes[first_item]->data->lowerBound + second_lb);
            else minlb = min(minlb, firstError + second_lb);
        }

        if (stopAfterError and depth == 0 and ub < FLT_MAX and *nodeError < ub) break;

    }

    if (similarlb) {
        similar_db1.free();
        similar_db2.free();
    }

    // we do not find the solution and the new lower bound is better than the old
    if (floatEqual(*nodeError, FLT_MAX)) node->data->lowerBound = max( node->data->lowerBound, max(ub, minlb) );

    Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", *nodeError);

    return {node, true};
}


void Search_cover_cache::run() {

    // Create empty list for candidate attributes
    Attributes attributes_to_visit;
    attributes_to_visit.reserve(GlobalParams::getInstance()->nattributes);

    // Update the candidate list based on frequency criterion
    if (minsup == 1) { // do not check frequency if minsup = 1
        for (int attr = 0; attr < GlobalParams::getInstance()->nattributes; ++attr) attributes_to_visit.push_back(attr);
    }
    else { // make sure each candidate attribute can be split into two nodes fulfilling the frequency criterion
        for (int attr = 0; attr < GlobalParams::getInstance()->nattributes; ++attr) {
            if (nodeDataManager->cover->temporaryIntersectSup(attr, false) >= minsup && nodeDataManager->cover->temporaryIntersectSup(attr) >= minsup)
                attributes_to_visit.push_back(attr);
        }
    }

    //create an empty array of items representing an emptyset and insert it
    Itemset itemset;
    Node * rootnode = cache->insert(nodeDataManager, 0).first;
//    Node * rootnode = cache->insert(nodeDataManager, 0, itemset).first;
    rootnode->data = nodeDataManager->initData();
    SimilarValss sdb1, sdb2;
    // call the recursive function to start the search
    cache->root = recurse(itemset, NO_ATTRIBUTE, rootnode, true, attributes_to_visit, 0, maxError, sdb1, sdb2).first;
}