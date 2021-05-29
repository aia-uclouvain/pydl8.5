#include "lcm_pruned.h"

using namespace std::chrono;


LcmPruned::LcmPruned(NodeDataManager *nodeDataManager, bool infoGain, bool infoAsc, bool repeatSort,
                     Support minsup,
                     Depth maxdepth,
                     Cache *cache,
                     int timeLimit,
                     bool continuous,
                     float maxError,
                     bool stopAfterError) :
        nodeDataManager(nodeDataManager), infoGain(infoGain), infoAsc(infoAsc), repeatSort(repeatSort),
        cache(cache),
        minsup(minsup),
        maxdepth(maxdepth),
        timeLimit(timeLimit),
        continuous(continuous),
        maxError(maxError),
        stopAfterError(stopAfterError)
{
    startTime = high_resolution_clock::now();
}

LcmPruned::~LcmPruned(){}

// the solution already exists for this node
Node *existingsolution(Node *node, Error *nodeError) {
    Logger::showMessageAndReturn("the solution exists and it is worth : ", *nodeError);
    return node;
}

// the node does not fullfil the constraints to be splitted (minsup, depth, etc.)
Node *cannotsplitmore(Node *node, Error ub, Error *nodeError, Error leafError) {
    Logger::showMessageAndReturn("max depth reached. ub = ", ub, " and leaf error = ", leafError);
    // we return the leaf error as node error without checking the upperbound constraint. The parent will do it
    *nodeError = leafError;
    return node;
}

// the node error is equal to the lower bound
Node *reachlowest(Node *node, Error *nodeError, Error leafError) {
    *nodeError = leafError;
    Logger::showMessageAndReturn("lowest error. node error = leaf error = ", *nodeError);
    return node;
}

// the upper bound of the node is lower than the lower bound
Node *infeasiblecase(Node *node, Error *saved_lb, Error ub) {
    Logger::showMessageAndReturn("no solution bcoz ub < lb. lb =", *saved_lb, " and ub = ", ub);
    return node;
}

Node * LcmPruned::getSolutionIfExists(Node *node, Error ub, Depth depth){
    Error *nodeError = &(((FND) node->data)->error);
    // in case the solution exists because the error of a newly created node is set to FLT_MAX
    if (*nodeError < FLT_MAX) {
        return existingsolution(node, nodeError);
    }

    Error *saved_lb = &(((FND) node->data)->lowerBound);
    // in case the problem is infeasible
    if (ub <= *saved_lb) {
        return infeasiblecase(node, saved_lb, ub);
    }

    Error leafError = ((FND) node->data)->leafError;
    // we reach the lowest value possible. implicitely, the upper bound constraint is not violated
    if (floatEqual(leafError, *saved_lb)) {
        return reachlowest(node, nodeError, leafError);
    }
//    nodeDataManager->cover->getSupport();

    // we cannot split tne node
    if (depth == maxdepth || nodeDataManager->cover->getSupport() < 2 * minsup) {
        return cannotsplitmore(node, ub, nodeError, leafError);
    }

    // if time limit is reached we backtrack
    if (timeLimitReached) {
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


Array<Attribute> LcmPruned::getSuccessors(Array<Attribute> last_candidates, Attribute last_added, Node* node) {

    std::multimap<float, Attribute> gain;
    Array<Attribute> next_candidates(last_candidates.size, 0);

    // the current node does not fullfill the frequency criterion. In correct situation, this case won't happen
    if (nodeDataManager->cover->getSupport() < 2 * minsup)
        return next_candidates;

    int current_sup = nodeDataManager->cover->getSupport();
    Supports current_sup_class = nodeDataManager->cover->getSupportPerClass();

    unordered_set<int> candidates_checker;
//    if (!infoGain){
//        for (TrieEdge edge : node->edges) {
//            candidates_checker.insert(item_attribute(edge.item));
//        }
//    }

    // access each candidate
    for (auto candidate : last_candidates) {

        // this attribute is already in the current itemset
        if (last_added == candidate) continue;

//        if (!infoGain && candidates_checker.find(candidate) != candidates_checker.end()){
//            next_candidates.push_back(candidate);
//            continue;
//        }

        // compute the support of each candidate
        int sup_left = nodeDataManager->cover->temporaryIntersectSup(candidate, false);
        int sup_right = current_sup - sup_left; //no need to intersect with negative item to compute its support

        // add frequent attributes but if heuristic is used to sort them, compute its value and sort later
        if (sup_left >= minsup && sup_right >= minsup) {
            //continuous dataset. Not supported yet
            if (continuous) {}
            else {
                if (infoGain) {
                    // compute the support per class in each split of the attribute to compute its IG value
                    Supports sup_class_left = nodeDataManager->cover->temporaryIntersect(candidate, false).first;
                    Supports sup_class_right = newSupports();
                    subSupports(current_sup_class, sup_class_left, sup_class_right);
                    gain.insert(std::pair<float, Attribute>(informationGain(sup_class_left, sup_class_right),
                                                            candidate));
                    deleteSupports(sup_class_left);
                    deleteSupports(sup_class_right);
                } else next_candidates.push_back(candidate);
            }
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
Error LcmPruned::computeSimilarityLowerBound(bitset<M> *b1_cover, bitset<M> *b2_cover, Error b1_error, Error b2_error) {
    return 0;
    if (is_python_error) return 0;
    Error bound = 0;
    bitset<M>*covers[] = {b1_cover, b2_cover};
    Error errors[] = {b1_error, b2_error};
    for (int i : {1}) {
        bitset<M>* cov = covers[i];
        Error err = errors[i];
        if (cov) {
            SupportClass sumdif = nodeDataManager->cover->countDif(cov);
            if (err - sumdif > bound) bound = err - sumdif;
        }
    }
    return (bound > 0) ? bound : 0;
}

// store the node with lowest error as well as the one with the largest cover in order to find a similarity lower bound
void LcmPruned::addInfoForLowerBound(NodeData *node_data, bitset<M> *&b1_cover, bitset<M> *&b2_cover,
                                    Error &b1_error, Error &b2_error, Support &highest_coversize) {
//    if (((FND) node_data)->error < FLT_MAX) {
    Error err = (((FND) node_data)->error < FLT_MAX) ? ((FND) node_data)->error : ((FND) node_data)->lowerBound;
    Support sup = nodeDataManager->cover->getSupport();

    if (err < FLT_MAX && err > b1_error) {
        delete[] b1_cover;
        b1_cover = nodeDataManager->cover->getTopBitsetArray();
        b1_error = err;
    }

    if (sup > highest_coversize) {
        delete[] b2_cover;
        b2_cover = nodeDataManager->cover->getTopBitsetArray();
        b2_error = err;
        highest_coversize = sup;
    }
//    }
}

static bool lte(const TrieEdge edge, const Item item) {
    return edge.item < item;
}

/** recurse - this method finds the best tree given an itemset and its cover and update
 * the information of the node representing the itemset. Each itemset is represented by a node and info about the
 * tree structure is wrapped into a variable data in the node object. Each itemset (the node) is inserted into the
 * trie (if it had not been inserted) before the call to the current function. When it has not been evaluated, the
 * data variable is set to null otherwise it contains info wrapped into FND object
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
Node *LcmPruned::recurse(Array<Item> itemset,
                             Attribute last_added,
                             Node *node,
                             Array<Attribute> next_candidates,
                             Depth depth,
                             float ub,
                             bool newnode) {

    // check if we ran out of time
    if (timeLimit > 0) {
        float runtime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count() / 1000.0;
        if (runtime >= timeLimit)
            timeLimitReached = true;
    }

    if (newnode) {
        Logger::showMessageAndReturn("New node");
        Logger::showMessageAndReturn("after init of the new node. leaf error = ", ((FND) node->data)->leafError);
        latticesize++;
    }
    else Logger::showMessageAndReturn("the node exists");
    Node* result = getSolutionIfExists(node, ub, depth);
//    cout << "haha " << result << endl;
    if (result) {
        if ( ((FND)node->data)->left && ((FND)node->data)->right ){
//                cout << "1" << endl;
Item leftI_down = item(((FND)node->data)->test, 0);
            TrieNode* left_down = (TrieNode*)cache->get(itemset, leftI_down);
//                cout << "2" << endl;
            Item rightI_down = item(((FND)node->data)->test, 1);
            TrieNode* right_down = (TrieNode*)cache->get(itemset, rightI_down);
//                cout << "3" << endl;
            Array<Item> copy_itemset;
            copy_itemset.duplicate(itemset);
            ///cout << "result found. update importances" << endl;
            ((TrieNode*) node)->changeImportance( left_down, right_down, copy_itemset, leftI_down, rightI_down, cache, true);
            copy_itemset.free();
            ///cout << "print loads after already found" << endl;
            vector<Item> v;
            ((Cache_Trie*)cache)->printload((TrieNode*)cache->root, v);
            cout << "end print load" << endl;
        }
//        ((TrieNode*) node)->changeImportance((TrieNode*)node, nullptr, itemset, attr, cache);
        return result;
    }
    Logger::showMessageAndReturn("Node solution cannot be found without calculation");

    // in case the solution cannot be derived without computation and remaining depth is 2, we use a specific algorithm
    /*if (nodeDataManager->maxdepth - depth == 2 && cover->getSupport() >= 2 * nodeDataManager->minsup && no_python_error) {
        return computeDepthTwo(ub, next_candidates, last_added, itemset, node, nodeDataManager, computed_lb, nodeDataManager->trie, this);
    }*/

    /* the node solution cannot be computed without calculation. at this stage, we will make a search through successors*/
    Error leafError = ((FND) node->data)->leafError;
    Error *nodeError = &(((FND) node->data)->error);
    Logger::showMessageAndReturn("leaf error = ", leafError, " new ub = ", ub);

    if (timeLimitReached) {
        *nodeError = leafError;
        return node;
    }

    // if we can't get solution without computation, we compute the next candidates to perform the search
    Array<Attribute> next_attributes = getSuccessors(next_candidates, last_added, node);
    // next_attributes = getSuccessors(next_candidates, cover, last_added);

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

    vector<Item> vec_items;
    vector<Node*> vec_nodes;
    vector<Node*> best_nodes;
    int best_attr;
    // we evaluate the split on each candidate attribute
    for(const auto attr : next_attributes) {
        Logger::showMessageAndReturn("\n\nWe are evaluating the attribute : ", attr);

        Array<Item> itemsets[2];
        Node *child_nodes[2];
        Error first_lb = -1, second_lb = -1;
        bool new_node;

        /* the lower bound is computed for both items. they are used as heuristic to decide
         the first item to branch on. We branch on item with higher lower bound to have chance
         to get a higher error to violate the ub constraint and prune the second branch
         this computation is costly so, in some case can add some overhead. If you don't
         want to use it, please comment the next block. 0/1 order is used in this case.*/

        //=========================== BEGIN BLOCK ==========================//
        nodeDataManager->cover->intersect(attr, false);
        first_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
        nodeDataManager->cover->backtrack();

        nodeDataManager->cover->intersect(attr);
        second_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
        nodeDataManager->cover->backtrack();
        //=========================== END BLOCK ==========================//


        first_item = second_lb > first_lb;
        second_item = !first_item;

        // perform search on the first item
        nodeDataManager->cover->intersect(attr, first_item);
        itemsets[first_item] = addItem(itemset, item(attr, first_item));
//        cout <<  "load: " << ((TrieNode*)node)->load << endl;
        pair<Node*, bool> node_state = cache->insert(itemsets[first_item], nodeDataManager);
        child_nodes[first_item] = node_state.first;
        new_node = node_state.second;
        // if lower bound was not computed
        if (floatEqual(first_lb, -1)) first_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
        // the best lower bound between the computed and the saved is used
        ((FND) child_nodes[first_item]->data)->lowerBound = (!new_node) ? max(((FND) child_nodes[first_item]->data)->lowerBound, first_lb) : first_lb;
        // perform the search for the first item
        child_nodes[first_item] = recurse(itemsets[first_item], attr, child_nodes[first_item], next_attributes,  depth + 1, child_ub, new_node);

        // check if the found information is relevant to compute the next similarity bounds
        addInfoForLowerBound(child_nodes[first_item]->data, b1_cover, b2_cover, b1_error, b2_error, highest_coversize);
        //cout << "after good bound 1" << " sc[0] = " << b1_sc[0] << " sc[1] = " << b1_sc[1] << " err = " << ((FND)nodes[first_item]->data)->error << endl;
        Error firstError = ((FND) child_nodes[first_item]->data)->error;
        itemsets[first_item].free();
        nodeDataManager->cover->backtrack();
        vec_items.push_back(item(attr, first_item));
        vec_nodes.push_back(child_nodes[first_item]);

        if (nodeDataManager->canimprove(child_nodes[first_item]->data, child_ub)) {
            // perform search on the second item
            nodeDataManager->cover->intersect(attr, second_item);
            itemsets[second_item] = addItem(itemset, item(attr, second_item));
//            cout <<  "load: " << ((TrieNode*)node)->load << endl;
            node_state = cache->insert(itemsets[second_item], nodeDataManager);
            child_nodes[second_item] = node_state.first;
            new_node = node_state.second;
            if (floatEqual(second_lb, -1)) second_lb = computeSimilarityLowerBound(b1_cover, b2_cover, b1_error, b2_error);
            // the best lower bound between the computed and the saved is used
            ((FND) child_nodes[second_item]->data)->lowerBound = (!new_node) ? max(((FND) child_nodes[second_item]->data)->lowerBound, second_lb) : second_lb;
            // bound for the second child (item)
            Error remainUb = child_ub - firstError;
            // perform the search for the second item
            child_nodes[second_item] = recurse(itemsets[second_item], attr, child_nodes[second_item], next_attributes, depth + 1, remainUb, new_node);

            // check if the found information is relevant to compute the next similarity bounds
            addInfoForLowerBound(child_nodes[second_item]->data, b1_cover, b2_cover, b1_error, b2_error, highest_coversize);
            Error secondError = ((FND) child_nodes[second_item]->data)->error;
            itemsets[second_item].free();
            nodeDataManager->cover->backtrack();
            vec_items.push_back(item(attr, second_item));
            vec_nodes.push_back(child_nodes[second_item]);

            Error feature_error = firstError + secondError;
//            printItemset(itemset);
//            if (itemset.size == 1 && itemset[0] == 0) cout << ((FND)((Cache_Hash*)cache)->bucket[2]->data)->leafError << endl;
//            cout << ((FND)node->data)->leafError << endl;
//            cout << ((FND)child_nodes[first_item]->data)->leafError << endl;
//            printItemset(itemset, true);
//            cout << "hereeez " << itemset.size << endl;
//            cout << node->data << endl;
//            cout << ((FND)node->data)->error << endl;

//            cout << "hohoh" << endl;
            TrieNode *old_left, *old_right;
            int old_attr;
            if ( !((FND)node->data)->left ) {
//                cout << "fi" << endl;
                old_left = nullptr;
                old_right = nullptr;
                old_attr = -1;
//                cout << "fa" << endl;
            }
            else {
//                cout << "fr" << endl;
                old_left = (TrieNode*)best_nodes[0];
                old_right = (TrieNode*)best_nodes[1];
                old_attr = best_attr;
//                old_left = lower_bound( ((TrieNode*)node)->edges.begin(), ((TrieNode*)node)->edges.end(), item(((FND)node->data)->test, 0), lte )->subtrie;
//                old_right = lower_bound( ((TrieNode*)node)->edges.begin(), ((TrieNode*)node)->edges.end(), item(((FND)node->data)->test, 1), lte )->subtrie;
//                cout << "gr" << endl;
            }
//            TrieNode *new_left = lower_bound( ((TrieNode*)node)->edges.begin(), ((TrieNode*)node)->edges.end(), item(attr, 0), lte )->subtrie;
//            TrieNode *new_right = lower_bound( ((TrieNode*)node)->edges.begin(), ((TrieNode*)node)->edges.end(), item(attr, 1), lte )->subtrie;
//            cout << "he" << endl;
TrieNode *new_left = (TrieNode*)child_nodes[0];
//            cout << "mo" << endl;
            TrieNode *new_right = (TrieNode*)child_nodes[1];
//            cout << "old new children" << endl;


//            printItemset(itemset, true);
//            cout << old_attr << endl;
            bool hasUpdated = nodeDataManager->updateData(node->data, child_ub, attr, child_nodes[0]->data, child_nodes[1]->data);
            if (hasUpdated) {
                child_ub = feature_error;
                best_nodes.clear();
                best_nodes.push_back(child_nodes[0]);
                best_nodes.push_back(child_nodes[1]);
                best_attr = attr;
//                cout << endl;
//                for (auto item: itemset) cout << item << ", ";
                Logger::showMessageAndReturn("-after this attribute ", attr, ", node error=", *nodeError, " and ub=", child_ub);
            }
            // in case we get the real error, we update the minimum possible error
            else minlb = min(minlb, feature_error);
            //printItemset(itemset, true);
            //cout << "current: " << ((FND)node->data)->test << " old: " << old << " hasupdated: " << hasUpdated << endl;
            //node->updateNode(((FND)node->data)->test, old, hasUpdated);

            Array<Item> copy_itemset;
            copy_itemset.duplicate(itemset);
            node->updateImportance(old_left, old_right, new_left, new_right, hasUpdated, copy_itemset, item(old_attr, 0), item(old_attr, 1), item(attr, 0), item(attr, 1), cache);
            copy_itemset.free();
            ///cout << "print loads after eval two items" << endl;
            vector<Item> v;
            ((Cache_Trie*)cache)->printload((TrieNode*)cache->root, v);
            ///cout << "end print load" << endl;
//            cout << "end up" << endl;

            if (nodeDataManager->canSkip(node->data)) { //lowerBound reached
                Logger::showMessageAndReturn("We get the best solution. So, we break the remaining attributes");
                break; //prune remaining attributes not browsed yet
            }
        } else { //we do not attempt the second child, so we use its lower bound
            // if the first error is unknown, we use its lower bound
            if (floatEqual(firstError, FLT_MAX)) minlb = min(minlb, first_lb + second_lb);
            // otherwise, we use it
            else minlb = min(minlb, firstError + second_lb);

            Array<Item> copy_itemset;
            copy_itemset.duplicate(itemset);
            ///cout << "Item " << first_item << " not good. change importance" << endl;
            ((TrieNode*) node)->changeImportance((TrieNode*)child_nodes[first_item], nullptr, copy_itemset, item(attr, first_item), -1, cache);
            copy_itemset.free();
            ///cout << "print loads after an item not good" << endl;
            vector<Item> v;
            ((Cache_Trie*)cache)->printload((TrieNode*)cache->root, v);
            ///cout << "end print load" << endl;
        }

        if (stopAfterError) {
            if (depth == 0 && ub < FLT_MAX) {
                if (*nodeError < ub)
                    break;
            }
        }
    }
    delete[] b1_cover;
    delete[] b2_cover;

    // we do not find the solution and the new lower bound is better than the old
    if (floatEqual(*nodeError, FLT_MAX) && max(ub, minlb) > ((FND) node->data)->lowerBound) ((FND) node->data)->lowerBound = max(ub, minlb);

    // update the load to free the cache later
//    node->update(vec_items, vec_nodes);

    /*// we do not get solution and new lower bound is better than the old
    Error *lb = &(((FND) node->data)->lowerBound);
    if (floatEqual(*nodeError, FLT_MAX) && max(ub, minlb) > *lb) {
        *lb = max(ub, minlb);
    }*/

    Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", *nodeError);

    next_attributes.free();
    return node;


}


void LcmPruned::run() {

    // Create empty list for candidate attributes
    Array<Attribute> attributes_to_visit(nattributes, 0);

    // Update the candidate list based on frequency criterion
    if (minsup == 1) { // do not check frequency if minsup = 1
        for (int attr = 0; attr < nattributes; ++attr) attributes_to_visit.push_back(attr);
    }
    else { // make sure each candidate attribute can be split into two nodes fulfilling the frequency criterion
        for (int attr = 0; attr < nattributes; ++attr) {
            if (nodeDataManager->cover->temporaryIntersectSup(attr, false) >= minsup && nodeDataManager->cover->temporaryIntersectSup(attr) >= minsup)
                attributes_to_visit.push_back(attr);
        }
    }

    //create an empty array of items representing an emptyset and insert it
    Array<Item> itemset;
    itemset.size = 0;
    itemset.elts = nullptr;

    // insert the emptyset node
    pair<Node *, bool> rootnode_state = cache->insert(itemset, nodeDataManager);

    // call the recursive function to start the search
    cache->root = recurse(itemset, NO_ATTRIBUTE, rootnode_state.first, attributes_to_visit, 0, maxError, rootnode_state.second);

    // never forget to return back what is not yours. Think to others who need it ;-)
    itemset.free();
    attributes_to_visit.free();

    /*cout << "ncall: " << ncall << endl;
    cout << "comptime: " << comptime << endl;
    cout << "searchtime: " << spectime << endl;
    cout << "totaltime: " << comptime + spectime << endl;
    ncall = 0; comptime = 0; spectime = 0;*/
}