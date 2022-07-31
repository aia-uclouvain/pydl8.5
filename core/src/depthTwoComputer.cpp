//
// Created by Gael Aglin on 26/09/2020.
//

#include "depthTwoComputer.h"
#include "search_base.h"
#include "cache_trie.h"

struct ErrorvalsDeleter {
    void operator()(const ErrorVals p) {
        delete[] p;
    }
};

void addTreeToCache(DepthTwo_NodeData* node_data,  const Itemset &itemset, Cache* cache){

    if (node_data->test >= 0) { // not a leaf

        Itemset itemset_left = addItem(itemset, item(node_data->test, NEG_ITEM));
        pair<Node *, bool> res_left = cache->insert(itemset_left);
        Node *node_left = res_left.first;
        if (res_left.second) { //new node. data is still null
            node_left->data = new TrieNodeData(*(((DepthTwo_NodeData*)node_data)->left));
        }
        else *((TrieNodeData*)node_left->data) = *(node_data->left);
        addTreeToCache(node_data->left, itemset_left, cache);

        Itemset itemset_right = addItem(itemset, item(node_data->test, POS_ITEM));
        pair<Node *, bool> res_right = cache->insert(itemset_right);
        Node *node_right = res_right.first;
        if (res_right.second) {
            node_right->data = new TrieNodeData(*(((DepthTwo_NodeData*)node_data)->right));
        }
        else *((TrieNodeData*)node_right->data) = *(node_data->right);
        addTreeToCache(node_data->right, itemset_right, cache);
    }
}

void addTreeToCache(Node* node, NodeDataManager* ndm, Cache* cache, Depth depth) {
    
    auto* node_data = (CoverNodeData*)node->data;

    if ( ((DepthTwo_NodeData*) (node_data->left)) != nullptr ) {

        ndm->cover->intersect(node_data->test, NEG_ITEM);
        pair<Node *, bool> res_left = cache->insert(ndm, depth);
        Node *node_left = res_left.first;
        if (res_left.second) {
           node_left->data = new CoverNodeData();
           *((CoverNodeData*)node_left->data) = *((DepthTwo_NodeData*)(node_data->left));
        }
        else *((CoverNodeData*)node_left->data) = *((DepthTwo_NodeData*)(node_data->left));
        node_data->left = (HashCoverNode*)node_left;
        addTreeToCache(node_left, ndm, cache, depth + 1);
        ndm->cover->backtrack();

        ndm->cover->intersect(node_data->test, POS_ITEM);
        pair<Node *, bool> res_right = cache->insert(ndm, depth);
        Node *node_right = res_right.first;
        if (res_right.second) {
            node_right->data = new CoverNodeData(*((DepthTwo_NodeData*)(node_data->right)));
        }
        else *((CoverNodeData*)node_right->data) = *((DepthTwo_NodeData*)(node_data->right));
        ((CoverNodeData*)node_data)->right = (HashCoverNode*)node_right;
        addTreeToCache(node_right, ndm, cache, depth + 1);
        ndm->cover->backtrack();
    }
}


/**
 * computeDepthTwo - this function compute the best tree given an itemset and the set of possible attributes
 * and return the root node of the found tree
 *
 * @param cover - the cover for which we are looking for an optimal tree
 * @param ub - the upper bound of the search
 * @param attributes_to_visit - the set of candidates attributes
 * @param last_added - the last attribute of item added before the search
 * @param itemset - the itemset at which point we are looking for the best tree
 * @param node - the node representing the itemset at which the best tree will be add
 * @param lb - the lower bound of the search
 * @return the same node passed as parameter is returned but the tree of depth 2 is already added to it
 */
Error computeDepthTwo(RCover* cover,
                      Error ub,
                      Attributes &attributes_to_visit,
                      Attribute last_added,
                      const Itemset &itemset,
                      Node *node,
                      NodeDataManager* nodeDataManager,
                      Error lb,
                      Cache* cache,
                      Search_base* searcher,
                      bool cachecover) {


   if (ub <= lb){ // infeasible case. Avoid computing useless solution
        node->data = nodeDataManager->initData(); // no need to update the error
        Logger::showMessageAndReturn("infeasible case. ub = ", ub, " lb = ", lb);
        return node->data->error;
    }

    // The fact to not bound the search make it find the best solution in any case and remove the chance to recall this
    // function for the same node with a higher upper bound. Since this function is not exponential, we can afford that.
    ub = FLT_MAX;

    // get the support and the support per class of the root node
    ErrorVals root_sup_clas = copyErrorVals(cover->getErrorValPerClass());
    Support root_sup = cover->getSupport();

    // update the next candidates list by removing the one already added
    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size() - 1);
    for(const auto attribute : attributes_to_visit) {
        if (last_added == attribute) continue;
        attr.push_back(attribute);
    }
    Logger::showMessageAndReturn("lowerbound ", lb);

    // compute the different support per class we need to perform the search. Only a few mandatory are computed. The remaining are derived from them
    // matrix for supports per class
    auto **sups_sc = new ErrorVals* [attr.size()];
    // matrix for support. In fact, for weighted examples problems, the sum of "support per class" is not equal to "support"
    auto **sups = new Support* [attr.size()];
    for (int i = 0; i < attr.size(); ++i) {
        // memory allocation
        sups_sc[i] = new ErrorVals[attr.size()];
        sups[i] = new Support[attr.size()];

        // compute values for first level of the tree
        cover->intersect(attr[i]);
        sups_sc[i][i] = copyErrorVals(cover->getErrorValPerClass());
        sups[i][i] = cover->getSupport();

        // compute value for second level
        for (int j = i + 1; j < attr.size(); ++j) {
            pair<ErrorVals, Support> p = cover->temporaryIntersect(attr[j]);
            sups_sc[i][j] = p.first;
            sups[i][j] = p.second;
        }
        // backtrack to recover the cover state
        cover->backtrack();
    }

    // the best tree we will have at the end
    unique_ptr<TreeTwo> best_tree(new TreeTwo());

    // find the best tree for each feature
    for (int i = 0; i < attr.size(); ++i) {
        Logger::showMessageAndReturn("root test: ", attr[i]);

        // best tree for the current feature
        unique_ptr<TreeTwo> feat_best_tree(new TreeTwo()); //here
        // set the root to the current feature
        feat_best_tree->root_data->test = attr[i];
        // compute its error and set it as initial error
        LeafInfo ev = nodeDataManager->computeLeafInfo();
        feat_best_tree->root_data->leafError = ev.error;

        ErrorVals idsc = sups_sc[i][i];
        Support ids = sups[i][i]; // Support ids = sumSupports(idsc);
        unique_ptr<ErrorVal, ErrorvalsDeleter> igsc(newErrorVals());
        subErrorVals(root_sup_clas, idsc, igsc.get());
        Support igs = root_sup - ids;

        // the feature tested as root is invalid since its two children cannot fulfill the minsup constraint
        if (igs < searcher->minsup or ids < searcher->minsup) {
            Logger::showMessageAndReturn("root impossible de splitter...on backtrack");
            continue; // test next root
        }

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //          LEFT CHILD       //
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//

        //find best feature to be left child of root
        feat_best_tree->root_data->left = new DepthTwo_NodeData();

        // the feature at root cannot be split at left. It is then a leaf node
        if (igs < 2 * searcher->minsup) {
            LeafInfo ev = nodeDataManager->computeLeafInfo(igsc.get());
            feat_best_tree->root_data->left->error = ev.error;
            feat_best_tree->root_data->left->test = (cachecover) ? ev.maxclass : -(ev.maxclass + 1);
            Logger::showMessageAndReturn("root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = ", feat_best_tree->root_data->left->error, " on backtrack");
            if (ev.error >= best_tree->root_data->error) {
                continue; // test next root
            }
        }
        // the root node can theoretically be split at left
        else {
            Logger::showMessageAndReturn("root gauche peut théoriquement spliter. Creusons plus...");
            Error feat_ub = best_tree->root_data->error; // the best tree found so far can be used as upper bound
            LeafInfo ev = nodeDataManager->computeLeafInfo(igsc.get());
            feat_best_tree->root_data->left->leafError = ev.error;
            feat_best_tree->root_data->left->test = (cachecover) ? ev.maxclass : -(ev.maxclass + 1);

            // explore different features to find the best left child
            for (int j = 0; j < attr.size(); ++j) {

                Logger::showMessageAndReturn("left test: ", attr[j]);

                if (attr[i] == attr[j]) {
                    Logger::showMessageAndReturn("left pareil que le parent ou non sup...on essaie un autre left");
                    continue; // test new left
                }

                ErrorVals jdsc = sups_sc[j][j], idjdsc = sups_sc[min(i, j)][max(i, j)];
                unique_ptr<ErrorVal, ErrorvalsDeleter> igjdsc(newErrorVals());
                subErrorVals(jdsc, idjdsc, igjdsc.get());
                Support jds = sups[j][j]; // Support jds = sumSupports(jdsc);
                Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                Support igjds = jds - idjds; // Support igjds =  sumSupports(igjdsc);
                Support igjgs = igs - igjds;

                // the feature tested at left can split and fulfill minsup contraint for its two children
                if (igjgs < searcher->minsup or igjds < searcher->minsup) {
                    Logger::showMessageAndReturn("le left testé ne peut splitter en pratique...un autre left!!!");
                    continue; // test new left
                }

                Logger::showMessageAndReturn("le left testé peut splitter. on le regarde");
                LeafInfo ev2 = nodeDataManager->computeLeafInfo(igjdsc.get());
                Logger::showMessageAndReturn("le left a droite produit une erreur de ", ev2.error);

                // upper bound constraint is violated
                if (ev2.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur gauche du left montre rien de bon. best root: ", best_tree->root_data->error, " error found: ", ev2.error, " Un autre left...");
                    continue; // test new left
                }

                unique_ptr<ErrorVal, ErrorvalsDeleter> igjgsc(newErrorVals());
                subErrorVals(igsc.get(), igjdsc.get(), igjgsc.get());
                LeafInfo ev1 = nodeDataManager->computeLeafInfo(igjgsc.get());
                Logger::showMessageAndReturn("le left a gauche produit une erreur de ", ev1.error);

                if (ev1.error + ev2.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur du left = ", ev1.error + ev2.error, " est pire que l'erreur du root existant (", feat_ub, "). Un autre left...");
                    continue; // test new left
                }

                // in case error found is equal to leaf error, we prefer a shallow tree
                if ( floatEqual(ev1.error + ev2.error, ev.error) ) {
                    feat_best_tree->root_data->left->error = ev.error;
                }
                else {
                    Logger::showMessageAndReturn("ce left ci donne une erreur sympa. On regarde a droite du root: ", ev1.error + ev2.error);

                    feat_best_tree->root_data->left->error = ev1.error + ev2.error;

                    if (feat_best_tree->root_data->left->left == nullptr){
                        feat_best_tree->root_data->left->left = new DepthTwo_NodeData(); //here
                        feat_best_tree->root_data->left->right = new DepthTwo_NodeData(); //here
                    }

                    feat_best_tree->root_data->left->left->error = ev1.error;
                    feat_best_tree->root_data->left->left->test = (cachecover) ? ev1.maxclass : -(ev1.maxclass + 1);
                    feat_best_tree->root_data->left->right->error = ev2.error;
                    feat_best_tree->root_data->left->right->test = (cachecover) ? ev2.maxclass : -(ev2.maxclass + 1);
                    feat_best_tree->root_data->left->test = attr[j];
                    feat_best_tree->root_data->left->size = 3;
                }

                feat_ub = ev1.error + ev2.error;

                if (floatEqual(ev1.error + ev2.error, 0)) {
                    break;
                }
            }

            // there is no left child coupled to the root to produce lower error than the best tree so far. No need to look at right
            if (floatEqual(feat_best_tree->root_data->left->error, FLT_MAX)){
                Logger::showMessageAndReturn("aucun left n'a su améliorer l'arbre existant: ", feat_best_tree->root_data->left->error, "on garde l'ancien arbre");
                continue; // test new root
            }
        }


        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //         RIGHT CHILD       //
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//

        //find best feature to be right child of root
        feat_best_tree->root_data->right = new DepthTwo_NodeData(); //here

        // the feature at root cannot be split at right. It is then a leaf node
        if (ids < 2 * searcher->minsup) {
            LeafInfo ev = nodeDataManager->computeLeafInfo(idsc);
            feat_best_tree->root_data->right->error = ev.error;
            feat_best_tree->root_data->right->test = (cachecover) ? ev.maxclass : -(ev.maxclass + 1);
            Logger::showMessageAndReturn("root droite ne peut théoriquement spliter; donc feuille. erreur droite = ", feat_best_tree->root_data->right->error, " on backtrack");
            if (ev.error >= best_tree->root_data->error - feat_best_tree->root_data->left->error) {
                continue; // test next root
            }
        }
        else { // the root node can theoretically be split at right
            Logger::showMessageAndReturn("root droite peut théoriquement spliter. Creusons plus...");
            Error feat_ub = best_tree->root_data->error - feat_best_tree->root_data->left->error;
            LeafInfo ev = nodeDataManager->computeLeafInfo(idsc);
            feat_best_tree->root_data->right->leafError = ev.error;
            feat_best_tree->root_data->right->test = (cachecover) ? ev.maxclass : -(ev.maxclass + 1);

            // in case we encounter the lower bound
            if (floatEqual(feat_best_tree->root_data->left->error + ev.error, lb)) {
                Logger::showMessageAndReturn("l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: ", feat_best_tree->root_data->right->error);
                feat_best_tree->root_data->right->error = ev.error;
                feat_best_tree->root_data->error = feat_best_tree->root_data->left->error + feat_best_tree->root_data->right->error;
                best_tree = move(feat_best_tree);
                break; // best is found
            }

            // explore different features to find the best right child
            for (int j = 0; j < attr.size(); ++j) {
                Logger::showMessageAndReturn("right test: ", attr[j]);

                if (attr[i] == attr[j]) {
                    Logger::showMessageAndReturn("right pareil que le parent ou non sup...on essaie un autre right");
                    continue; // test next right
                }

                ErrorVals idjdsc = sups_sc[min(i, j)][max(i, j)];
                unique_ptr<ErrorVal, ErrorvalsDeleter> idjgsc(newErrorVals());
                subErrorVals(idsc, idjdsc, idjgsc.get());
                Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                Support idjgs = ids - idjds; // Support idjgs = sumSupports(idjgsc);

                // the feature tested at right can split and fulfill minsup contraint for its two children
                if (idjgs < searcher->minsup or idjds < searcher->minsup) {
                    Logger::showMessageAndReturn("le right testé ne peut splitter...un autre right!!!");
                    continue; // test next right
                }

                Logger::showMessageAndReturn("le right testé peut splitter. on le regarde");
                LeafInfo ev1 = nodeDataManager->computeLeafInfo(idjgsc.get());
                Logger::showMessageAndReturn("le right a gauche produit une erreur de ", ev1.error);

                if (ev1.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur gauche du right montre rien de bon. Un autre right...");
                    continue; // test next right
                }

                LeafInfo ev2 = nodeDataManager->computeLeafInfo(idjdsc);
                Logger::showMessageAndReturn("le right a droite produit une erreur de ", ev2.error);

                if (ev1.error + ev2.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur du right = ", ev1.error + ev2.error, " n'ameliore pas l'existant. Un autre right...");
                    continue; // test next right
                }

                // in case error found is equal to leaf error, we prefer a shallow tree
                if ( floatEqual(ev1.error + ev2.error, ev.error) ) {
                    feat_best_tree->root_data->right->error = ev.error;
                }
                else {
                    Logger::showMessageAndReturn("ce right ci donne une meilleure erreur que les précédents right: ", ev1.error + ev2.error);

                    feat_best_tree->root_data->right->error = ev1.error + ev2.error;

                    if (feat_best_tree->root_data->right->left == nullptr) {
                        feat_best_tree->root_data->right->left = new DepthTwo_NodeData();
                        feat_best_tree->root_data->right->right = new DepthTwo_NodeData();
                    }

                    feat_best_tree->root_data->right->left->error = ev1.error;
                    feat_best_tree->root_data->right->left->test = (cachecover) ? ev1.maxclass : -(ev1.maxclass + 1);
                    feat_best_tree->root_data->right->right->error = ev2.error;
                    feat_best_tree->root_data->right->right->test = (cachecover) ? ev2.maxclass : -(ev2.maxclass + 1);
                    feat_best_tree->root_data->right->test = attr[j];
                    feat_best_tree->root_data->right->size = 3;
                }

                feat_ub = ev1.error + ev2.error;

                if (floatEqual(feat_best_tree->root_data->right->error + feat_best_tree->root_data->left->error, lb) or floatEqual(ev1.error + ev2.error, 0)) {
                    break;
                }
            }

            // there is no left child coupled to the root and left to produce lower error than the best tree so far.
            if (floatEqual(feat_best_tree->root_data->right->error, FLT_MAX)){
                Logger::showMessageAndReturn("pas d'arbre mieux que le meilleur jusque là.");
                continue; // test new root
            }

            feat_best_tree->root_data->error = feat_best_tree->root_data->left->error + feat_best_tree->root_data->right->error;
            feat_best_tree->root_data->size = feat_best_tree->root_data->left->size + feat_best_tree->root_data->right->size + 1;
            best_tree = move(feat_best_tree);
            Logger::showMessageAndReturn("ce triplet (root, left, right) ci donne une meilleure erreur que les précédents triplets: (", best_tree->root_data->test, ",", best_tree->root_data->left->left ? best_tree->root_data->left->test : -best_tree->root_data->left->test, ",", best_tree->root_data->right->left ? best_tree->root_data->right->test : -best_tree->root_data->right->test, ") err:", best_tree->root_data->error);
            if (floatEqual(best_tree->root_data->error, lb)) {
                Logger::showMessageAndReturn("The best tree is found");
                break;
            }
        }
    }

    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) deleteErrorVals(sups_sc[k][i]);
        delete [] sups_sc[k];
        delete [] sups[k];
    }
    delete [] sups_sc;
    delete [] sups;
    deleteErrorVals(root_sup_clas);

    if (best_tree->root_data and best_tree->root_data->left and best_tree->root_data->right) Logger::showMessageAndReturn("root: ", best_tree->root_data->test, " left: ", best_tree->root_data->left->test, " right: ", best_tree->root_data->right->test);

    // without cache
    if (node == nullptr && cache == nullptr){
        Error best_error = best_tree->root_data->error;
        return best_error;
    }

    if (best_tree->root_data->test != INT32_MAX) {

        Logger::showMessageAndReturn("best tree found (root, left, right): (", best_tree->root_data->test, ",", best_tree->root_data->left->left ? best_tree->root_data->left->test : -best_tree->root_data->left->test, ",", best_tree->root_data->right->left ? best_tree->root_data->right->test : -best_tree->root_data->right->test, ") err:", best_tree->root_data->error);

        if (best_tree->root_data->size == 3 and best_tree->root_data->left->test == best_tree->root_data->right->test and floatEqual(best_tree->root_data->leafError, best_tree->root_data->left->error + best_tree->root_data->right->error)) {
            best_tree->root_data->size = 1;
            best_tree->root_data->error = best_tree->root_data->leafError;
            best_tree->root_data->test = best_tree->root_data->right->test;
            Logger::showMessageAndReturn("best twotree error = ", to_string(best_tree->root_data->error));
            if (cachecover){
                *((CoverNodeData*)node->data) = *(best_tree->root_data);
                ((CoverNodeData*)node->data)->left = nullptr;
                ((CoverNodeData*)node->data)->right = nullptr;
            }
            else {
                *((TrieNodeData*)node->data) = *(best_tree->root_data);
            }
            return node->data->error;
        }

        if (cachecover) {
            *((CoverNodeData*)node->data) = *(best_tree->root_data);
            addTreeToCache(node, nodeDataManager, cache, searcher->maxdepth - 2 + 1);
        }
        else {
            if (cache->maxcachesize > NO_CACHE_LIMIT and cache->getCacheSize() + ((searcher->maxdepth + 1) * 4) > cache->maxcachesize) {
                cache->wipe();
            }
            *((TrieNodeData*)node->data) = *(best_tree->root_data);
            addTreeToCache(best_tree->root_data, itemset, cache);
            Logger::showMessageAndReturn("tre: ", node->data->test);
        }
        Logger::showMessageAndReturn("best twotree error = ", to_string(node->data->error));
        return node->data->error;
    } else {
        node->data->error = node->data->leafError;
        return node->data->error;
    }

}