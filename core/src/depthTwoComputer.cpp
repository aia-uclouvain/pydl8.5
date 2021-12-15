//
// Created by Gael Aglin on 26/09/2020.
//

#include "depthTwoComputer.h"
#include "search_base.h"
#include "cache_trie.h"
//#include "search_nocache.h"


//void addTreeToCache(Node* node, const Itemset &itemset, Cache* cache){

void addTreeToCache(Node* node,  Itemset &itemset, Cache* cache){
    auto* node_data = node->data;
    if (node_data->left and node_data->left->data) {

        Itemset itemset_left = addItem(itemset, item(node_data->test, 0));
        Node *node_left = cache->insert(itemset_left).first;
        node_left->data = node_data->left->data;
        node_data->left->data = nullptr;
        delete node_data->left;
        node_data->left = node_left;
//        ((TrieNode*)node_left)->search_parents.push_back((TrieNode*)node);
//        ((TrieNode*)node_left)->search_parents.insert(make_pair((TrieNode*)node, itemset));
        addTreeToCache(node_left, itemset_left, cache);

//        if(itemset_left.size() == 5 and itemset_left.at(0) == 16 and itemset_left.at(1) == 26 and itemset_left.at(2) == 40 and itemset_left.at(3) == 43 and itemset_left.at(4) == 46) {
//            cout << "\nparchild*(";
//            printItemset(itemset, true, true);
//            cout << node_data->test << " " << node_data->leafError << " " << node_data->error << endl;
//            printItemset(itemset_left, true);
//            exit(0);
//        }

//        if(itemset_left.size() == 5 and itemset_left.at(0) == 16 and itemset_left.at(1) == 28 and itemset_left.at(2) == 40 and itemset_left.at(3) == 43 and itemset_left.at(4) == 46) {
//            cout << "\nparchild*(";
//            printItemset(itemset, true, true);
//            cout << node_data->test << " " << node_data->leafError << " " << node_data->error << endl;
//            printItemset(itemset_left, true);
//            exit(0);
//        }

        Itemset itemset_right = addItem(itemset, item(node_data->test, 1));
        Node *node_right = cache->insert(itemset_right).first;
        node_right->data = node_data->right->data;
        node_data->right->data = nullptr;
        delete node_data->right;
        node_data->right = node_right;
//        ((TrieNode*)node_right)->search_parents.push_back((TrieNode*)node);
//        ((TrieNode*)node_right)->search_parents.insert(make_pair((TrieNode*)node, itemset));
        addTreeToCache(node_right, itemset_right, cache);

//        if(itemset_right.size() == 5 and itemset_right.at(0) == 16 and itemset_right.at(1) == 28 and itemset_right.at(2) == 40 and itemset_right.at(3) == 43 and itemset_right.at(4) == 46) {
//            cout << "\nparchild-(";
//            printItemset(itemset, true, true);
//            cout << node_data->test << endl;
//            printItemset(itemset_right, true);
//            exit(0);
//        }

//        if (node_data->test >= 0) node_data->test *= -1;
//        cache->updateParents(node, node_left, node_right);
        cache->updateParents(node, node_left, node_right, itemset);
//        node_data->test *= -1;
    }
}

void addTreeToCache(Node* node, NodeDataManager* ndm, Cache* cache){
    auto* node_data = node->data;
    if (node_data->left){
        ndm->cover->intersect(node_data->test, NEG_ITEM);
        Node *node_left = cache->insert(ndm).first;
        node_left->data = node_data->left->data;
        addTreeToCache(node_left, ndm, cache);
        ndm->cover->backtrack();

        ndm->cover->intersect(node_data->test, POS_ITEM);
        Node *node_right = cache->insert(ndm).first;
        node_right->data = (NodeData *) node_data->right->data;
        addTreeToCache(node_left, ndm, cache);
        ndm->cover->backtrack();
    }
}

void setIteme(NodeData* node_data, const Itemset & itemset, Cache* cache){
    if (node_data->left){
        Itemset itemset_left(itemset.size() + 1);
        addItem(itemset, item(((FND)node_data->left->data)->test, 0), itemset_left);
        Node *node_left = cache->insert(itemset_left).first;
        node_left->data = (NodeData *) node_data->left;
        setIteme(node_left->data, itemset_left, cache);
//        itemset_left.free();
    }

    if (node_data->right){
        Itemset itemset_right(itemset.size() + 1);
        addItem(itemset, item(((FND)node_data->right->data)->test, 1), itemset_right);
        Node *node_right = cache->insert(itemset_right).first;
        node_right->data = (NodeData *) node_data->right;
        setIteme(node_right->data, itemset_right, cache);
//        itemset_right.free();
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
                      Itemset &itemset,
                      Node *node,
                      NodeDataManager* nodeDataManager,
                      Error lb,
                      Cache* cache,
                      Search_base* searcher,
                      bool cachecover) {

    // infeasible case. Avoid computing useless solution
    if (ub <= lb){
        node->data = nodeDataManager->initData(); // no need to update the error
//        if (verbose) cout << "infeasible case. ub = " << ub << " lb = " << lb << endl;
        Logger::showMessageAndReturn("infeasible case. ub = ", ub, " lb = ", lb);
        return ((FND)node->data)->error;
    }

    // The fact to not bound the search make it find the best solution in any case and remove the chance to recall this
    // function for the same node with an higher upper bound. Since this function is not exponential, we can afford that.
    ub = FLT_MAX;
//    cout << "popo" << endl;

    //initialize the timer to count the time spent in this function
    auto start = high_resolution_clock::now();

    //local variable to make the function verbose or not. Can be improved :-)
//    bool local_verbose = verbose;

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
//    if(itemset.size() ==3 and itemset.at(0) == 16 and itemset.at(1) == 43 and itemset.at(2) == 46){
//        verbose = true;
//        if (std::find(attr.begin(), attr.end(), 22) == attr.end())
//            attr.push_back(22);
//    }

    // compute the different support per class we need to perform the search
    // only a few mandatory are computed. The remaining are derived from them
    auto start_comp = high_resolution_clock::now();
    // matrix for supports per class
    auto **sups_sc = new ErrorVals*[attr.size()];
    // matrix for support. In fact, for weighted examples problems, the sum of "support per class" is not equal to "support"
    auto **sups = new Support* [attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        // memory allocation
        sups_sc[l] = new ErrorVals[attr.size()];
        sups[l] = new Support[attr.size()];

        // compute values for first level of the tree
        cover->intersect(attr[l]);
        sups_sc[l][l] = copyErrorVals(cover->getErrorValPerClass());
        sups[l][l] = cover->getSupport();

        // compute value for second level
        for (int i = l + 1; i < attr.size(); ++i) {
            pair<ErrorVals, Support> p = cover->temporaryIntersect(attr[i]);
            sups_sc[l][i] = p.first;
            sups[l][i] = p.second;
        }
        // backtrack to recover the cover state
        cover->backtrack();
    }

    auto* best_tree = new TreeTwo();

    // find the best tree for each feature
    for (int i = 0; i < attr.size(); ++i) {
//        if (local_verbose) cout << "root test: " << attr[i] << endl;
        Logger::showMessageAndReturn("root test: ", attr[i]);

        // best tree for the current feature
        auto* feat_best_tree = new TreeTwo();
        // set the root to the current feature
        feat_best_tree->root_data->test = attr[i];
        // compute its error and set it as initial error
        LeafInfo ev = nodeDataManager->computeLeafInfo();
        feat_best_tree->root_data->leafError = ev.error;

        ErrorVals idsc = sups_sc[i][i];
        Support ids = sups[i][i]; // Support ids = sumSupports(idsc);
        ErrorVals igsc = newErrorVals();
        subErrorVals(root_sup_clas, idsc, igsc);
        Support igs = root_sup - ids;

        //feature to left.
        // the feature cannot be root since its two children will not fulfill the minsup constraint
        if (igs < searcher->minsup || ids < searcher->minsup) {
//            if (local_verbose) cout << "root impossible de splitter...on backtrack" << endl;
            Logger::showMessageAndReturn("root impossible de splitter...on backtrack");
            delete feat_best_tree;
            deleteErrorVals(igsc);
            continue;
        }

        feat_best_tree->root_data->left = new Node();
        feat_best_tree->root_data->left->data = new Freq_NodeData();
        feat_best_tree->root_data->right = new Node();
        feat_best_tree->root_data->right->data = new Freq_NodeData();

        // the feature at root cannot be split at left. It is then a leaf node
        if (igs < 2 * searcher->minsup) {
            LeafInfo ev = nodeDataManager->computeLeafInfo(igsc);
            feat_best_tree->root_data->left->data->error = ev.error;
            feat_best_tree->root_data->left->data->test = ev.maxclass;
//            if (local_verbose) cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << feat_best_tree->root_data->left->error << " on backtrack" << endl;
            Logger::showMessageAndReturn("root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = ", feat_best_tree->root_data->left->data->error, " on backtrack");
        }
        // the root node can theorically be split at left
        else {
//            if (local_verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            Logger::showMessageAndReturn("root gauche peut théoriquement spliter. Creusons plus...");
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            LeafInfo ev = nodeDataManager->computeLeafInfo(igsc);
            feat_best_tree->root_data->left->data->error = min(ev.error, best_tree->root_data->error);
            feat_best_tree->root_data->left->data->leafError = ev.error;
            feat_best_tree->root_data->left->data->test = ev.maxclass;

            if (!floatEqual(ev.error, lb)) {
                Error tmp = feat_best_tree->root_data->left->data->error;
                for (int j = 0; j < attr.size(); ++j) {
//                    if (local_verbose) cout << "left test: " << attr[j] << endl;
                    Logger::showMessageAndReturn("left test: ", attr[j]);
                    if (attr[i] == attr[j]) {
//                        if (local_verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        Logger::showMessageAndReturn("left pareil que le parent ou non sup...on essaie un autre left");
                        continue;
                    }
                    ErrorVals jdsc = sups_sc[j][j], idjdsc = sups_sc[min(i, j)][max(i, j)], igjdsc = newErrorVals();
                    subErrorVals(jdsc, idjdsc, igjdsc);
                    Support jds = sups[j][j]; // Support jds = sumSupports(jdsc);
                    Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                    Support igjds = jds - idjds; // Support igjds =  sumSupports(igjdsc);
                    Support igjgs = igs - igjds;

                    // the root node can in practice be split into two children
                    if (igjgs >= searcher->minsup && igjds >= searcher->minsup) {
//                        if (local_verbose) cout << "le left testé peut splitter. on le regarde" << endl;
                        Logger::showMessageAndReturn("le left testé peut splitter. on le regarde");

                        LeafInfo ev2 = nodeDataManager->computeLeafInfo(igjdsc);
//                        if (local_verbose) cout << "le left a droite produit une erreur de " << ev2.error << endl;
                        Logger::showMessageAndReturn("le left a droite produit une erreur de ", ev2.error);

                        if (ev2.error >= min(best_tree->root_data->error, feat_best_tree->root_data->left->data->error)) {
//                            if (local_verbose) cout << "l'erreur gauche du left montre rien de bon. best root: " << best_tree->root_data->error << " best left: " << feat_best_tree->root_data->left->error << " Un autre left..." << endl;
                            Logger::showMessageAndReturn("l'erreur gauche du left montre rien de bon. best root: ", best_tree->root_data->error, " best left: ", feat_best_tree->root_data->left->data->error, " Un autre left...");
                            deleteErrorVals(igjdsc);
                            continue;
                        }

                        ErrorVals igjgsc = newErrorVals();
                        subErrorVals(igsc, igjdsc, igjgsc);
                        LeafInfo ev1 = nodeDataManager->computeLeafInfo(igjgsc);
//                        if (local_verbose) cout << "le left a gauche produit une erreur de " << ev1.error << endl;
                        Logger::showMessageAndReturn("le left a gauche produit une erreur de ", ev1.error);

                        if (ev1.error + ev2.error < min(best_tree->root_data->error, feat_best_tree->root_data->left->data->error)) {
                            feat_best_tree->root_data->left->data->error = ev1.error + ev2.error;
//                            if (local_verbose) cout << "ce left ci donne une meilleure erreur que les précédents left: " << feat_best_tree->root_data->left->error << endl;
                            Logger::showMessageAndReturn("ce left ci donne une meilleure erreur que les précédents left: ", feat_best_tree->root_data->left->data->error);
                            if (!feat_best_tree->root_data->left->data->left){
                                feat_best_tree->root_data->left->data->left = new Node();
                                feat_best_tree->root_data->left->data->left->data = new Freq_NodeData();
                                feat_best_tree->root_data->left->data->right = new Node();
                                feat_best_tree->root_data->left->data->right->data = new Freq_NodeData();
                            }

                            feat_best_tree->root_data->left->data->left->data->error = ev1.error;
                            feat_best_tree->root_data->left->data->left->data->test = ev1.maxclass;
                            feat_best_tree->root_data->left->data->right->data->error = ev2.error;
                            feat_best_tree->root_data->left->data->right->data->test = ev2.maxclass;
                            feat_best_tree->root_data->left->data->test = attr[j];
                            feat_best_tree->root_data->left->data->size = 3;

//                            if (floatEqual(feat_best_tree->root_data->left->data->error, lb)) {
//                                deleteErrorVals(igjdsc);
//                                deleteErrorVals(igjgsc);
//                                break;
//                            }
                        } else {
//                            if (local_verbose) cout << "l'erreur du left = " << ev1.error + ev2.error << " n'ameliore pas l'existant. Un autre left..." << endl;
                            Logger::showMessageAndReturn("l'erreur du left = ", ev1.error + ev2.error, " n'ameliore pas l'existant. Un autre left...");
                        }
                        deleteErrorVals(igjgsc);
//                    } else if (local_verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                    } else Logger::showMessageAndReturn("le left testé ne peut splitter en pratique...un autre left!!!");
                    deleteErrorVals(igjdsc);
                }
                if (floatEqual(feat_best_tree->root_data->left->data->error, tmp)){
                    // do not use the best tree error but the feat left leaferror
                    feat_best_tree->root_data->left->data->error = feat_best_tree->root_data->left->data->leafError;
//                    if (local_verbose) cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: " << feat_best_tree->root_data->left->error << endl;
                    Logger::showMessageAndReturn("aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: ", feat_best_tree->root_data->left->data->error);
                }
            } else {
//                if (local_verbose) cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << feat_best_tree->root_data->left->error << endl;
                Logger::showMessageAndReturn("l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: ", feat_best_tree->root_data->left->data->error);
            }
        }


        //feature to right
        if (feat_best_tree->root_data->left->data->error < best_tree->root_data->error) {
//            if (local_verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;
            Logger::showMessageAndReturn("vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite");

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * searcher->minsup) {
                LeafInfo ev = nodeDataManager->computeLeafInfo(idsc);
                feat_best_tree->root_data->right->data->error = ev.error;
                feat_best_tree->root_data->right->data->test = ev.maxclass;
//                if (local_verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << feat_best_tree->root_data->right->error << " on backtrack" << endl;
                Logger::showMessageAndReturn("root droite ne peut théoriquement spliter; donc feuille. erreur droite = ", feat_best_tree->root_data->right->data->error, " on backtrack");
            } else {
//                if (local_verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                Logger::showMessageAndReturn("root droite peut théoriquement spliter. Creusons plus...");
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                LeafInfo ev = nodeDataManager->computeLeafInfo(idsc);
                Error remainingError = best_tree->root_data->error - feat_best_tree->root_data->left->data->error;
                feat_best_tree->root_data->right->data->error = min(ev.error, remainingError);
                feat_best_tree->root_data->right->data->leafError = ev.error;
                feat_best_tree->root_data->right->data->test = ev.maxclass;

                Error tmp = feat_best_tree->root_data->right->data->error;

                if (!floatEqual(ev.error, lb)) {
                    for (int j = 0; j < attr.size(); ++j) {
//                        if (local_verbose) cout << "right test: " << attr[j] << endl;
                        Logger::showMessageAndReturn("right test: ", attr[j]);
                        if (attr[i] == attr[j]) {
//                            if (local_verbose) cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            Logger::showMessageAndReturn("right pareil que le parent ou non sup...on essaie un autre right");
                            continue;
                        }

                        ErrorVals idjdsc = sups_sc[min(i, j)][max(i, j)], idjgsc = newErrorVals();
                        subErrorVals(idsc, idjdsc, idjgsc);
                        Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                        Support idjgs = ids - idjds; // Support idjgs = sumSupports(idjgsc);

                        // the root node can in practice be split into two children
                        if (idjgs >= searcher->minsup && idjds >= searcher->minsup) {
//                            if (local_verbose) cout << "le right testé peut splitter. on le regarde" << endl;
                            Logger::showMessageAndReturn("le right testé peut splitter. on le regarde");
                            LeafInfo ev1 = nodeDataManager->computeLeafInfo(idjgsc);
//                            if (local_verbose) cout << "le right a gauche produit une erreur de " << ev1.error << endl;
                            Logger::showMessageAndReturn("le right a gauche produit une erreur de ", ev1.error);

                            if (ev1.error >= min(remainingError, feat_best_tree->root_data->right->data->error)) {
//                                if (local_verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                Logger::showMessageAndReturn("l'erreur gauche du right montre rien de bon. Un autre right...");
                                deleteErrorVals(idjgsc);
                                continue;
                            }

                            LeafInfo ev2 = nodeDataManager->computeLeafInfo(idjdsc);
//                            if (local_verbose) cout << "le right a droite produit une erreur de " << ev2.error << endl;
                            Logger::showMessageAndReturn("le right a droite produit une erreur de ", ev2.error);
                            if (ev1.error + ev2.error < min(remainingError, feat_best_tree->root_data->right->data->error)) {
                                feat_best_tree->root_data->right->data->error = ev1.error + ev2.error;
//                                if (local_verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << feat_best_tree->root_data->right->error << endl;
                                Logger::showMessageAndReturn("ce right ci donne une meilleure erreur que les précédents right: ", feat_best_tree->root_data->right->data->error);
                                if (!feat_best_tree->root_data->right->data->left){
                                    feat_best_tree->root_data->right->data->left = new Node();
                                    feat_best_tree->root_data->right->data->left->data = new Freq_NodeData();
                                    feat_best_tree->root_data->right->data->right = new Node();
                                    feat_best_tree->root_data->right->data->right->data = new Freq_NodeData();
                                }

                                feat_best_tree->root_data->right->data->left->data->error = ev1.error;
                                feat_best_tree->root_data->right->data->left->data->test = ev1.maxclass;
                                feat_best_tree->root_data->right->data->right->data->error = ev2.error;
                                feat_best_tree->root_data->right->data->right->data->test = ev2.maxclass;
                                feat_best_tree->root_data->right->data->test = attr[j];
                                feat_best_tree->root_data->right->data->size = 3;

                                if (floatEqual(feat_best_tree->root_data->right->data->error + feat_best_tree->root_data->left->data->error, lb)) {
                                    deleteErrorVals(idjgsc);
                                    break;
                                }
                            } else {
//                                if (local_verbose) cout << "l'erreur du right = " << ev1.error + ev2.error << " n'ameliore pas l'existant. Un autre right..." << endl;
                                Logger::showMessageAndReturn("l'erreur du right = ", ev1.error + ev2.error, " n'ameliore pas l'existant. Un autre right...");
                            }
//                        } else if (local_verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        } else Logger::showMessageAndReturn("le right testé ne peut splitter...un autre right!!!");
                        deleteErrorVals(idjgsc);
                    }
                    if (floatEqual(feat_best_tree->root_data->right->data->error, tmp)){
                        // in this case, do not use the remaining as error but leaferror
                        feat_best_tree->root_data->right->data->error = feat_best_tree->root_data->right->data->leafError;
//                        if (local_verbose) cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: " << feat_best_tree->root_data->right->error << endl;
                        Logger::showMessageAndReturn("aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: ", feat_best_tree->root_data->right->data->error);
                    }
//                } else if (local_verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << feat_best_tree->root_data->right->error << endl;
                } else Logger::showMessageAndReturn("l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: ", feat_best_tree->root_data->right->data->error);
            }

            if (feat_best_tree->root_data->left->data->error + feat_best_tree->root_data->right->data->error < best_tree->root_data->error) {
                feat_best_tree->root_data->error = feat_best_tree->root_data->left->data->error + feat_best_tree->root_data->right->data->error;
                feat_best_tree->root_data->size += feat_best_tree->root_data->left->data->size + feat_best_tree->root_data->right->data->size;

                best_tree->replaceTree(feat_best_tree);
//                if (local_verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << best_tree->root_data->error << " " << best_tree->root_data->test << endl;
                Logger::showMessageAndReturn("ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: ", best_tree->root_data->error, " ", best_tree->root_data->test);
            } else {
                delete feat_best_tree;
//                if (local_verbose) cout << "cet arbre n'est pas mieux que le meilleur jusque là." << endl;
                Logger::showMessageAndReturn("cet arbre n'est pas mieux que le meilleur jusque là.");
            }
        }
        else delete feat_best_tree;
        deleteErrorVals(igsc);

    }

    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) {
            deleteErrorVals(sups_sc[k][i]);
        }
        delete [] sups_sc[k];
        delete [] sups[k];
    }
    delete [] sups_sc;
    delete [] sups;
    deleteErrorVals(root_sup_clas);
//    if (local_verbose && best_tree->root_data && best_tree->root_data->left && best_tree->root_data->right) cout << "root: " << best_tree->root_data->test << " left: " << best_tree->root_data->left->test << " right: " << best_tree->root_data->right->test << endl;
    if (best_tree->root_data && best_tree->root_data->left && best_tree->root_data->right) Logger::showMessageAndReturn("root: ", best_tree->root_data->test, " left: ", best_tree->root_data->left->data->test, " right: ", best_tree->root_data->right->data->test);
//    if (local_verbose) cout << "le1: " << best_tree->root_data->left->left->error << " le2: " << best_tree->root_data->left->right->error << " re1: " << best_tree->root_data->right->left->error << " re2: " << best_tree->root_data->right->right->error << endl;
//    if (local_verbose) cout << "ble: " << best_tree->root_data->left->error << " bre: " << best_tree->root_data->right->error << " broe: " << best_tree->root_data->error << endl;
//    if (local_verbose) cout << "lc1: " << best_tree->root_data->left->left->test << " lc2: " << best_tree->root_data->left->right->test << " rc1: " << best_tree->root_data->right->left->test << " rc2: " << best_tree->root_data->right->right->test << endl;
//    if (local_verbose) cout << "blc: " << best_tree->root_data->left->test << " brc: " << best_tree->root_data->right->test << endl;

    // without cache
    if (node == nullptr && cache == nullptr){
        Error best_error = best_tree->root_data->error;
        delete best_tree;
        return best_error;
    }

    if (best_tree->root_data->test != -1) {
        if (best_tree->root_data->size == 3 && best_tree->root_data->left->data->test == best_tree->root_data->right->data->test && floatEqual(best_tree->root_data->leafError, best_tree->root_data->left->data->error + best_tree->root_data->right->data->error)) {
            best_tree->root_data->size = 1;
            best_tree->root_data->error = best_tree->root_data->leafError;
            best_tree->root_data->test = best_tree->root_data->right->data->test;
            delete best_tree->root_data->left;
            best_tree->root_data->left = nullptr;
            delete best_tree->root_data->right;
            best_tree->root_data->right = nullptr;
//            if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->error) << endl;
            Logger::showMessageAndReturn("best twotree error = ", to_string(best_tree->root_data->error));
            node->data = (NodeData *)best_tree->root_data;
            best_tree->root_data = nullptr;
            delete best_tree;
            return node->data->error;
        }

        //delete node->data;

        if (cachecover) {
            node->data = (NodeData *) best_tree->root_data;
            best_tree->root_data = nullptr;
            addTreeToCache(node, nodeDataManager, cache);
        }
        else {
//            if (cache->maxcachesize > NO_CACHE_LIMIT and cache->getCacheSize() + best_tree->root_data->size - 1 > cache->maxcachesize) {
//            cout << cache->getCacheSize() << " " << cache->getCacheSize() + ((searcher->maxdepth + 1) * 4) << " " << cache->maxcachesize << endl;
            if (cache->maxcachesize > NO_CACHE_LIMIT and cache->getCacheSize() + ((searcher->maxdepth + 1) * 4) > cache->maxcachesize) {
//                cout << "wipe_in" << endl;
                cache->wipe();
            }
//            cout << endl;
            node->data = (NodeData *) best_tree->root_data;
            best_tree->root_data = nullptr;
//            cout << "ite:";
//            printItemset(itemset, true);
//            printItemset(attributes_to_visit, true);
//            printItemset(attr, true);
//            cout << node->data->error << endl;
//            if(itemset.size() ==3 and itemset.at(0) == 16 and itemset.at(1) == 43 and itemset.at(2) == 46){
//                verbose = false;
//            }
            addTreeToCache(node, itemset, cache);
//            cout << endl;
        }

//        if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->error) << endl;
        Logger::showMessageAndReturn("best twotree error = ", to_string(node->data->error));

        delete best_tree;
//        best_tree->free();
        return node->data->error;
    } else {
        //error not lower than ub (this case will never happen as the ub is set to FLT_MAX)
        //it can happen if ub = FLT_MAX and no successor can split
//        cout << "pas possible" << endl;
//        delete best_tree;
        best_tree->free();
        node->data->error = node->data->leafError;
        return node->data->error;
    }

}