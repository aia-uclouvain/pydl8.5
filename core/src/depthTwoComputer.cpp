//
// Created by Gael Aglin on 26/09/2020.
//

#include "depthTwoComputer.h"
#include "rCoverTotalFreq.h"


// If we are using this computer, we are not optimising quantiles --> nquantiles == 1
void setItem(QueryData_Best* node_data, Array<Item> itemset, Trie* trie){
    if (node_data->lefts[0]){
        Array<Item> itemset_left;
        itemset_left.alloc(itemset.size + 1);
        //cout << node_data->lefts[0]->tests[0] << endl;
        addItem(itemset, item(node_data->lefts[0]->tests[0], 0), itemset_left);
        TrieNode *node_left = trie->insert(itemset_left);
        node_left->data = (QueryData *) node_data->lefts[0];
        setItem((QueryData_Best *)node_left->data, itemset_left, trie);
        itemset_left.free();
    }

    if (node_data->rights[0]){
        Array<Item> itemset_right;
        itemset_right.alloc(itemset.size + 1);
        addItem(itemset, item(node_data->rights[0]->tests[0], 1), itemset_right);
        TrieNode *node_right = trie->insert(itemset_right);
        node_right->data = (QueryData *) node_data->rights[0];
        setItem((QueryData_Best *)node_right->data, itemset_right, trie);
        itemset_right.free();
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
TrieNode* computeDepthTwo( RCover* cover,
                           Error ub,
                           Array <Attribute> attributes_to_visit,
                           Attribute last_added,
                           Array <Item> itemset,
                           TrieNode *node,
                           Query* query,
                           Error lb,
                           Trie* trie) {

    // infeasible case. Avoid computing useless solution
    if (ub <= lb){
        node->data = query->initData(cover); // no need to update the error
        if (verbose) cout << "infeasible case. ub = " << ub << " lb = " << lb << endl;
        return node;
    }
//    cout << "fifi" << endl;

    // The fact to not bound the search make it find the best solution in any case and remove the chance to recall this
    // function for the same node with an higher upper bound. Since this function is not exponential, we can afford that.
    ub = FLT_MAX;

    //count the number of call to this function for stats
    ncall += 1;
    //initialize the timer to count the time spent in this function
    auto start = high_resolution_clock::now();

    //local variable to make the function verbose or not. Can be improved :-)
    bool local_verbose = verbose;
//    if (last_added == 64) local_verbose = true;

    // get the support and the support per class of the root node
    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    Support root_sup = cover->getSupport();

    // update the next candidates list by removing the one already added
    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for(const auto& attribute : attributes_to_visit) {
        if (last_added == attribute) continue;
        attr.push_back(attribute);
    }

    // compute the different support per class we need to perform the search
    // only a few mandatory are computed. The remaining are derived from them
    auto start_comp = high_resolution_clock::now();
    // matrix for supports per class
    auto **sups_sc = new Supports *[attr.size()];
    // matrix for support. In fact, for weighted examples problems, the sum of "support per class" is not equal to "support"
    auto **sups = new Support* [attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        // memory allocation
        sups_sc[l] = new Supports[attr.size()];
        sups[l] = new Support[attr.size()];

        // compute values for first level of the tree
//        cout << "item : " << attr[l] << " ";
        cover->intersect(attr[l]);
        sups_sc[l][l] = copySupports(cover->getSupportPerClass());
        sups[l][l] = cover->getSupport();

        // compute value for second level
        for (int i = l + 1; i < attr.size(); ++i) {
//            cout << "\titem_fils : " << attr[i] << " ";
            pair<Supports, Support> p = cover->temporaryIntersect(attr[i]);
            sups_sc[l][i] = p.first;
            sups[l][i] = p.second;
        }
        // backtrack to recover the cover state
        cover->backtrack();
    }
    auto stop_comp = high_resolution_clock::now();
    comptime += duration<double>(stop_comp - start_comp).count();

    auto* best_tree = new TreeTwo(cover->dm->getNQuantiles());
    //TreeTwo* feat_best_tree;

    // find the best tree for each feature
    for (int i = 0; i < attr.size(); ++i) {
        if (local_verbose) cout << "root test: " << attr[i] << endl;
        //cout << "beeest " << best_tree->root_data->errors[0] << endl;

        // best tree for the current feature
        auto* feat_best_tree = new TreeTwo(cover->dm->getNQuantiles());
//        cout << "beeest " << best_tree->root_data->errors[0] << endl;
        // set the root to the current feature
        feat_best_tree->root_data->tests[0] = attr[i];
        // compute its error and set it as initial error
        LeafInfo ev = query->computeLeafInfo(cover);
        feat_best_tree->root_data->leafErrors[0] = ev.error;
        // feat_best_tree.root_data->tests[0] = ev.maxclass;

        Supports idsc = sups_sc[i][i];
        Support ids = sups[i][i]; // Support ids = sumSupports(idsc);
        Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc);
        Support igs = root_sup - ids;

        //feature to left
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup) {
            if (local_verbose) cout << "root impossible de splitter...on backtrack" << endl;
            delete feat_best_tree;
            deleteSupports(igsc);
            continue;
        }

        feat_best_tree->root_data->lefts[0] = new QueryData_Best(cover->dm->getNQuantiles());
        feat_best_tree->root_data->rights[0] = new QueryData_Best(cover->dm->getNQuantiles());

        // the feature at root cannot be splitted at left. It is then a leaf node
        if (igs < 2 * query->minsup) {
            LeafInfo ev = query->computeLeafInfo(igsc);
            feat_best_tree->root_data->lefts[0]->errors[0] = ev.error;
            feat_best_tree->root_data->lefts[0]->tests[0] = ev.maxclass;
            if (local_verbose)
                cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << feat_best_tree->root_data->lefts[0]->errors[0] << " on backtrack" << endl;
        }
        // the root node can theorically be split at left
        else {
            if (local_verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
//            cout << "beeest " << best_tree->root_data->errors[0] << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            LeafInfo ev = query->computeLeafInfo(igsc);
            feat_best_tree->root_data->lefts[0]->errors[0] = min(ev.error, best_tree->root_data->errors[0]);
            feat_best_tree->root_data->lefts[0]->leafErrors[0] = ev.error;
            feat_best_tree->root_data->lefts[0]->tests[0] = ev.maxclass;

            if (!floatEqual(ev.error, lb)) {
                Error tmp = feat_best_tree->root_data->lefts[0]->errors[0];
                for (int j = 0; j < attr.size(); ++j) {
                    if (local_verbose) cout << "left test: " << attr[j] << endl;
                    if (attr[i] == attr[j]) {
                        if (local_verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }
                    Supports jdsc = sups_sc[j][j], idjdsc = sups_sc[min(i, j)][max(i, j)], igjdsc = newSupports();
                    subSupports(jdsc, idjdsc, igjdsc);
                    Support jds = sups[j][j]; // Support jds = sumSupports(jdsc);
                    Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                    Support igjds = jds - idjds; // Support igjds =  sumSupports(igjdsc);
                    Support igjgs = igs - igjds;

                    // the root node can in practice be split into two children
                    if (igjgs >= query->minsup && igjds >= query->minsup) {
                        if (local_verbose) cout << "le left testé peut splitter. on le regarde" << endl;
//                        cout << "beeest " << best_tree->root_data->errors[0] << endl;

                        LeafInfo ev2 = query->computeLeafInfo(igjdsc);
                        if (local_verbose) cout << "le left a droite produit une erreur de " << ev2.error << endl;
//                        cout << "beeest " << best_tree->root_data->errors[0] << endl;

                        if (ev2.error >= min(best_tree->root_data->errors[0], feat_best_tree->root_data->lefts[0]->errors[0])) {
                            if (local_verbose)
                                cout << "l'erreur gauche du left montre rien de bon. best root: " << best_tree->root_data->errors[0] << " best left: " << feat_best_tree->root_data->lefts[0]->errors[0] << " Un autre left..." << endl;
                            deleteSupports(igjdsc);
                            continue;
                        }

                        Supports igjgsc = newSupports();
                        subSupports(igsc, igjdsc, igjgsc);
                        LeafInfo ev1 = query->computeLeafInfo(igjgsc);
                        if (local_verbose) cout << "le left a gauche produit une erreur de " << ev1.error << endl;
//                        cout << "beeest " << best_tree->root_data->errors[0] << endl;

                        if (ev1.error + ev2.error < min(best_tree->root_data->errors[0], feat_best_tree->root_data->lefts[0]->errors[0])) {
                            feat_best_tree->root_data->lefts[0]->errors[0] = ev1.error + ev2.error;
                            if (local_verbose)
                                cout << "ce left ci donne une meilleure erreur que les précédents left: " << feat_best_tree->root_data->lefts[0]->errors[0] << endl;
                            if (!feat_best_tree->root_data->lefts[0]->lefts[0]){
                                feat_best_tree->root_data->lefts[0]->lefts[0] = new QueryData_Best(cover->dm->getNQuantiles());
                                feat_best_tree->root_data->lefts[0]->rights[0] = new QueryData_Best(cover->dm->getNQuantiles());
                            }
                            //else feat_best_tree.cleanLeft();
                            feat_best_tree->root_data->lefts[0]->lefts[0]->errors[0] = ev1.error;
                            feat_best_tree->root_data->lefts[0]->lefts[0]->tests[0] = ev1.maxclass;
                            feat_best_tree->root_data->lefts[0]->rights[0]->errors[0] = ev2.error;
                            feat_best_tree->root_data->lefts[0]->rights[0]->tests[0] = ev2.maxclass;
                            feat_best_tree->root_data->lefts[0]->tests[0] = attr[j];
//                            feat_best_tree.root_data->lefts[0]->lefts[0] = feat_best_tree.left1_data;
//                            feat_best_tree.root_data->lefts[0]->rights[0] = feat_best_tree.left2_data;
                            feat_best_tree->root_data->lefts[0]->sizes[0] = 3;

                            if (floatEqual(feat_best_tree->root_data->lefts[0]->errors[0], lb)) {
                                deleteSupports(igjdsc);
                                deleteSupports(igjgsc);
                                break;
                            }
                        } else {
                            if (local_verbose)
                                cout << "l'erreur du left = " << ev1.error + ev2.error << " n'ameliore pas l'existant. Un autre left..." << endl;
                        }
//                        cout << "beeest " << best_tree->root_data->errors[0] << endl;
                        deleteSupports(igjgsc);
                    } else if (local_verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                    deleteSupports(igjdsc);
                }
                if (floatEqual(feat_best_tree->root_data->lefts[0]->errors[0], tmp)){
                    // do not use the best tree error but the feat left leaferror
                    feat_best_tree->root_data->lefts[0]->errors[0] = feat_best_tree->root_data->lefts[0]->leafErrors[0];
                    if (local_verbose) cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: " << feat_best_tree->root_data->lefts[0]->errors[0] << endl;
                }
            } else {
                if (local_verbose)
                    cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << feat_best_tree->root_data->lefts[0]->errors[0] << endl;
            }
        }


        //feature to right
//        cout << "bestoor si error " << best_tree->root_data->errors[0] << endl;
        if (feat_best_tree->root_data->lefts[0]->errors[0] < best_tree->root_data->errors[0]) {
            if (local_verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * query->minsup) {
                LeafInfo ev = query->computeLeafInfo(idsc);
                feat_best_tree->root_data->rights[0]->errors[0] = ev.error;
                feat_best_tree->root_data->rights[0]->tests[0] = ev.maxclass;
                if (local_verbose)
                    cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << feat_best_tree->root_data->rights[0]->errors[0] << " on backtrack" << endl;
            } else {
                if (local_verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                LeafInfo ev = query->computeLeafInfo(idsc);
                Error remainingError = best_tree->root_data->errors[0] - feat_best_tree->root_data->lefts[0]->errors[0];
                feat_best_tree->root_data->rights[0]->errors[0] = min(ev.error, remainingError);
                feat_best_tree->root_data->rights[0]->leafErrors[0] = ev.error;
                feat_best_tree->root_data->rights[0]->tests[0] = ev.maxclass;

                Error tmp = feat_best_tree->root_data->rights[0]->errors[0];

                if (!floatEqual(ev.error, lb)) {
                    for (int j = 0; j < attr.size(); ++j) {
                        if (local_verbose) cout << "right test: " << attr[j] << endl;
                        if (attr[i] == attr[j]) {
                            if (local_verbose)
                                cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }

                        Supports idjdsc = sups_sc[min(i, j)][max(i, j)], idjgsc = newSupports();
                        subSupports(idsc, idjdsc, idjgsc);
                        Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                        Support idjgs = ids - idjds; // Support idjgs = sumSupports(idjgsc);

                        // the root node can in practice be split into two children
                        if (idjgs >= query->minsup && idjds >= query->minsup) {
                            if (local_verbose) cout << "le right testé peut splitter. on le regarde" << endl;
                            LeafInfo ev1 = query->computeLeafInfo(idjgsc);
                            if (local_verbose) cout << "le right a gauche produit une erreur de " << ev1.error << endl;

                            if (ev1.error >= min(remainingError, feat_best_tree->root_data->rights[0]->errors[0])) {
                                if (local_verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                deleteSupports(idjgsc);
                                continue;
                            }

                            LeafInfo ev2 = query->computeLeafInfo(idjdsc);
                            if (local_verbose) cout << "le right a droite produit une erreur de " << ev2.error << endl;
                            if (ev1.error + ev2.error < min(remainingError, feat_best_tree->root_data->rights[0]->errors[0])) {
                                feat_best_tree->root_data->rights[0]->errors[0] = ev1.error + ev2.error;
                                if (local_verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << feat_best_tree->root_data->rights[0]->errors[0] << endl;
                                if (!feat_best_tree->root_data->rights[0]->lefts[0]){
                                    feat_best_tree->root_data->rights[0]->lefts[0] = new QueryData_Best(cover->dm->getNQuantiles());
                                    feat_best_tree->root_data->rights[0]->rights[0] = new QueryData_Best(cover->dm->getNQuantiles());
                                }
                                //else feat_best_tree.removeRight();
                                feat_best_tree->root_data->rights[0]->lefts[0]->errors[0] = ev1.error;
                                feat_best_tree->root_data->rights[0]->lefts[0]->tests[0] = ev1.maxclass;
                                feat_best_tree->root_data->rights[0]->rights[0]->errors[0] = ev2.error;
                                feat_best_tree->root_data->rights[0]->rights[0]->tests[0] = ev2.maxclass;
                                feat_best_tree->root_data->rights[0]->tests[0] = attr[j];
//                                feat_best_tree.right_data->lefts[0] = feat_best_tree.right1_data;
//                                feat_best_tree.right_data->rights[0] = feat_best_tree.right2_data;
                                feat_best_tree->root_data->rights[0]->sizes[0] = 3;

                                if (floatEqual(feat_best_tree->root_data->rights[0]->errors[0], lb)) {
                                    deleteSupports(idjgsc);
                                    break;
                                }
                            } else {
                                if (local_verbose) cout << "l'erreur du right = " << ev1.error + ev2.error << " n'ameliore pas l'existant. Un autre right..." << endl;
                            }
                        } else if (local_verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        deleteSupports(idjgsc);
                    }
                    if (floatEqual(feat_best_tree->root_data->rights[0]->errors[0], tmp)){
                        // in this case, do not use the remaining as error but leaferror
                        feat_best_tree->root_data->rights[0]->errors[0] = feat_best_tree->root_data->rights[0]->leafErrors[0];
                        if (local_verbose) cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: " << feat_best_tree->root_data->rights[0]->errors[0] << endl;
                    }
                } else if (local_verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << feat_best_tree->root_data->rights[0]->errors[0] << endl;
            }

            if (feat_best_tree->root_data->lefts[0]->errors[0] + feat_best_tree->root_data->rights[0]->errors[0] < best_tree->root_data->errors[0]) {
                feat_best_tree->root_data->errors[0] = feat_best_tree->root_data->lefts[0]->errors[0] + feat_best_tree->root_data->rights[0]->errors[0];
                feat_best_tree->root_data->sizes[0] += feat_best_tree->root_data->lefts[0]->sizes[0] + feat_best_tree->root_data->rights[0]->sizes[0];

                //best_tree = feat_best_tree;
                //cout << "replaccc" << endl;
                best_tree->replaceTree(feat_best_tree);
                if (local_verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << best_tree->root_data->errors[0] << " " << best_tree->root_data->tests[0] << endl;
            } else {
                delete feat_best_tree;
                if (local_verbose) cout << "cet arbre n'est pas mieux que le meilleur jusque là." << endl;
            }
        }
        else delete feat_best_tree;
        deleteSupports(igsc);

        //if (feat_best_tree && best_tree->root_data != feat_best_tree->root_data) delete feat_best_tree;
    }
//    cout << "ffffi" << endl;
    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) {
            deleteSupports(sups_sc[k][i]);
        }
        delete [] sups_sc[k];
        delete [] sups[k];
    }
    delete [] sups_sc;
    delete [] sups;
    deleteSupports(root_sup_clas);
    if (local_verbose && best_tree->root_data && best_tree->root_data->lefts[0] && best_tree->root_data->rights[0]) cout << "root: " << best_tree->root_data->tests[0] << " left: " << best_tree->root_data->lefts[0]->tests[0] << " right: " << best_tree->root_data->rights[0]->tests[0] << endl;
//    if (local_verbose) cout << "le1: " << best_tree->root_data->lefts[0]->lefts[0]->errors[0] << " le2: " << best_tree->root_data->lefts[0]->rights[0]->errors[0] << " re1: " << best_tree->root_data->rights[0]->lefts[0]->errors[0] << " re2: " << best_tree->root_data->rights[0]->rights[0]->errors[0] << endl;
//    if (local_verbose) cout << "ble: " << best_tree->root_data->lefts[0]->errors[0] << " bre: " << best_tree->root_data->rights[0]->errors[0] << " broe: " << best_tree->root_data->errors[0] << endl;
//    if (local_verbose) cout << "lc1: " << best_tree->root_data->lefts[0]->lefts[0]->tests[0] << " lc2: " << best_tree->root_data->lefts[0]->rights[0]->tests[0] << " rc1: " << best_tree->root_data->rights[0]->lefts[0]->tests[0] << " rc2: " << best_tree->root_data->rights[0]->rights[0]->tests[0] << endl;
//    if (local_verbose) cout << "blc: " << best_tree->root_data->lefts[0]->tests[0] << " brc: " << best_tree->root_data->rights[0]->tests[0] << endl;

    if (best_tree->root_data->tests[0] != -1) {
        if (best_tree->root_data->sizes[0] == 3 && best_tree->root_data->lefts[0]->tests[0] == best_tree->root_data->rights[0]->tests[0] && floatEqual(best_tree->root_data->leafErrors[0], best_tree->root_data->lefts[0]->errors[0] + best_tree->root_data->rights[0]->errors[0])) {
            best_tree->root_data->sizes[0] = 1;
            best_tree->root_data->errors[0] = best_tree->root_data->leafErrors[0];
            best_tree->root_data->tests[0] = best_tree->root_data->rights[0]->tests[0];
            delete best_tree->root_data->lefts[0];
            best_tree->root_data->lefts[0] = nullptr;
            delete best_tree->root_data->rights[0];
            best_tree->root_data->rights[0] = nullptr;
            node->data = (QueryData *)best_tree->root_data;
            auto stop = high_resolution_clock::now();
            spectime += duration<double>(stop - stop_comp).count();
            if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->errors[0]) << endl;
            return node;
        }

        node->data = (QueryData *) best_tree->root_data;
        setItem((QueryData_Best *) node->data, itemset, trie);

        auto stop = high_resolution_clock::now();
        spectime += duration<double>(stop - stop_comp).count();

        if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->errors[0]) << endl;
        return node;
    } else {
        //error not lower than ub
        LeafInfo ev = query->computeLeafInfo(cover);
        delete best_tree;
        node->data = (QueryData *) new QueryData_Best(cover->dm->getNQuantiles());
        ((QueryData_Best *) node->data)->errors[0] = ev.error;
        ((QueryData_Best *) node->data)->leafErrors[0] = ev.error;
        ((QueryData_Best *) node->data)->tests[0] = ev.maxclass;
        auto stop = high_resolution_clock::now();
        spectime += duration<double>(stop - stop_comp).count();
        // if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->errors[0]) << endl;

        return node;
    }

}