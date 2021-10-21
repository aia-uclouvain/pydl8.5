//
// Created by Gael Aglin on 19/10/2021.
//

#include "search_nocache.h"

using namespace std::chrono;


Search_nocache::Search_nocache(NodeDataManager *nodeDataManager, bool infoGain, bool infoAsc, bool repeatSort,
                               Support minsup,
                               Depth maxdepth,
                               int timeLimit,
                               float maxError,
                               bool specialAlgo,
                               bool stopAfterError,
                               bool use_ub) :
        Search_base(nodeDataManager, infoGain, infoAsc, repeatSort, minsup, maxdepth, timeLimit, maxError, specialAlgo, stopAfterError), use_ub(use_ub) {
    startTime = high_resolution_clock::now();
}

Search_nocache::~Search_nocache() {}


// information gain calculation
float Search_nocache::informationGain(Supports notTaken, Supports taken) {
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


Array<Attribute> Search_nocache::getSuccessors(Array<Attribute> last_candidates, Attribute last_added) {

    std::multimap<float, Attribute> gain;
    Array<Attribute> next_candidates(last_candidates.size, 0);

    int current_sup = nodeDataManager->cover->getSupport();
    Supports current_sup_class = nodeDataManager->cover->getSupportPerClass();

    // access each candidate
    for (const auto &candidate: last_candidates) {

        // this attribute is already in the current itemset
        if (last_added == candidate) continue;

        // compute the support of each candidate
        int sup_left = nodeDataManager->cover->temporaryIntersectSup(candidate, false);
        int sup_right = current_sup - sup_left; //no need to intersect with negative item to compute its support

        // add frequent attributes but if heuristic is used to sort them, compute its value and sort later
        if (sup_left >= minsup && sup_right >= minsup) {
            if (not infoGain) next_candidates.push_back(candidate);
            else {
                // compute the support per class in each split of the attribute to compute its IG value
                Supports sup_class_left = nodeDataManager->cover->temporaryIntersect(candidate, false).first;
                Supports sup_class_right = newSupports();
                subSupports(current_sup_class, sup_class_left, sup_class_right);
                gain.insert(std::pair<float, Attribute>(informationGain(sup_class_left, sup_class_right), candidate));
                deleteSupports(sup_class_left);
                deleteSupports(sup_class_right);
            }
        }
    }

    // if heuristic is used, add the next candidates given the heuristic order
    if (infoGain) {
        if (infoAsc) for (auto &it: gain) next_candidates.push_back(it.second); //items with low IG first
        else
            for (auto it = gain.rbegin(); it != gain.rend(); ++it)
                next_candidates.push_back(it->second); //items with high IG first
    }
    // disable the heuristic variable if the sort must be performed once
    if (!repeatSort) infoGain = false;

    return next_candidates;
}

/** recurse - this method finds the best tree given an itemset and its cover and update
 * the information of the node representing the itemset. Each itemset is represented by a node and info about the
 * tree structure is wrapped into a variable data in the node object. Each itemset (the node) is inserted into the
 * trie (if it had not been inserted) before the call to the current function. When it has not been evaluated, the
 * data variable is set to null otherwise it contains info wrapped into FND object
 *
 * @param last_added - the last added attribute
 * @param next_candidates - next attributes to visit
 * @param cover - the transactions covered by the itemset
 * @param depth - the current depth in the search tree
 * @param ub - the upper bound of the search. It cannot be reached
 * @return the same node as get in parameter with added information about the best tree
 */
Error Search_nocache::recurse(Attribute last_added,
                              Array<Attribute> next_candidates,
                              Depth depth,
                              float ub) {

    // check if we ran out of time
    if (timeLimit > 0 && duration<float>(high_resolution_clock::now() - startTime).count() >= (float)timeLimit) timeLimitReached = true;

    // if upper bound is disabled, we set it to infinity
    if (not use_ub) ub = FLT_MAX;

    auto leaf = nodeDataManager->computeLeafInfo();

    // the solution can be inferred without computation
    if (depth == maxdepth || nodeDataManager->cover->getSupport() < 2 * minsup || leaf.error == 0 || timeLimitReached) {
        Logger::showMessageAndReturn("we backtrack with leaf error = ", leaf.error, " new ub = ", ub);
        return leaf.error;
    }

    Logger::showMessageAndReturn("Node solution cannot be found without calculation");

    // in case the solution cannot be derived without computation and remaining depth is 2, we use a specific algorithm
    if (specialAlgo && maxdepth - depth == 2 && nodeDataManager->cover->getSupport() >= 2 * minsup && no_python_error) {
        return computeDepthTwo(nodeDataManager->cover, ub, next_candidates, last_added, Array<Item>(), nullptr, nodeDataManager, 0, nullptr, this).second;
    }

    /* the node solution cannot be computed without calculation. at this stage, we will make a search through successors*/
    Logger::showMessageAndReturn("leaf error = ", leaf.error, " new ub = ", ub);

    if (timeLimitReached) return leaf.error;

    // if we can't get solution without computation, we compute the next candidates to perform the search
    Array<Attribute> next_attributes = getSuccessors(next_candidates, last_added);
    // next_attributes = getSuccessors(next_candidates, cover, last_added);

    // case in which there is no candidate
    if (next_attributes.size == 0) {
        Logger::showMessageAndReturn("No candidates. nodeError is set to leafError");
        Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", leaf.error);
        Logger::showMessageAndReturn("we backtrack with leaf error ", leaf.error);
        next_attributes.free();
        return leaf.error;
    }

    bool first_item = false, second_item = true;
    Error best_error = leaf.error;

    // we evaluate the split on each candidate attribute
    for (const auto &attr: next_attributes) {
        Logger::showMessageAndReturn("\n\nWe are evaluating the attribute : ", attr);

        Logger::showMessageAndReturn("Item left");
        nodeDataManager->cover->intersect(attr, first_item);
        Error child_ub = ub;
        Error firstError = recurse(attr, next_attributes, depth + 1, child_ub);
        nodeDataManager->cover->backtrack();

        if (firstError >= child_ub) {
            Logger::showMessageAndReturn("left violate constraint. error = ", firstError, " ub was = ", ub);
            continue;
        }

        Logger::showMessageAndReturn("Item right");
        nodeDataManager->cover->intersect(attr, second_item);
        child_ub = child_ub - firstError;
        Error secondError = recurse(attr, next_attributes, depth + 1, child_ub);
        nodeDataManager->cover->backtrack();

        Error feature_error = firstError + secondError;
        if (feature_error < best_error) {
            best_error = feature_error;
            ub = feature_error;
            Logger::showMessageAndReturn("-after this attribute ", attr, ", node error=", best_error, " and ub=", ub);
            if (best_error == 0) { //lowerBound reached
                Logger::showMessageAndReturn("We get the best solution. So, we break the remaining attributes");
                break; //prune remaining attributes not browsed yet
            }
        }
        else Logger::showMessageAndReturn("error found is high = ", feature_error, " best was = ", best_error);

        if (stopAfterError && depth == 0 && ub < FLT_MAX && best_error < ub) break;
    }

    Logger::showMessageAndReturn("depth = ", depth, " and init ub = ", ub, " and error after search = ", best_error);

    next_attributes.free();
    return best_error;


}


void Search_nocache::run() {

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

    // call the recursive function to start the search
    Error tree_error = recurse(NO_ATTRIBUTE, attributes_to_visit, 0, maxError);

    if (use_ub) cout << "upper bound is used" << endl;
    else cout << "upper bound is not used" << endl;
    cout << "tree error = " << tree_error << endl;

    // never forget to return what is not yours. Think to others who need it ;-)
    itemset.free();
    attributes_to_visit.free();
}