//
// Created by Gael Aglin on 26/09/2020.
//

#include "depthTwoComputer.h"
#include "rCoverTotalFreq.h"

void setItem(QueryData_Best* node_data, Array<Item> itemset, Trie* trie){
    if (node_data->left){
        Array<Item> itemset_left;
        itemset_left.alloc(itemset.size + 1);
        addItem(itemset, item(node_data->left->test, 0), itemset_left);
        TrieNode *node_left = trie->insert(itemset_left);
        node_left->data = (QueryData *) node_data->left;
        setItem((QueryData_Best *)node_left->data, itemset_left, trie);
        itemset_left.free();
    }

    if (node_data->right){
        Array<Item> itemset_right;
        itemset_right.alloc(itemset.size + 1);
        addItem(itemset, item(node_data->right->test, 1), itemset_right);
        TrieNode *node_right = trie->insert(itemset_right);
        node_right->data = (QueryData *) node_data->right;
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
TrieNode* computeDepthTwo(RCover* cover,
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
        return node;
    }

    // The fact to not bound the search make it find the best solution in any case and remove the chance to recall this
    // function for the same node with an higher upper bound. Since this function is not exponential, we can afford that.
    ub = FLT_MAX;

    //count the number of call to this function for stats
    ncall += 1;
    //initialize the timer to count the time spent in this function
    auto start = high_resolution_clock::now();

    //local variable to make the function verbose or not. Can be improved :-)
    bool verbose = false;

    // get the support and the support per class of the root node
    Supports root_sup_clas = copySupports(cover->getSupportPerClass(query->weights));
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
    Supports **sups_sc = new Supports *[attr.size()];
    Support **sups = new Support* [attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
//        cout << "i = " << l << endl;
        sups_sc[l] = new Supports[attr.size()];
        sups[l] = new Support[attr.size()];
        /*((RCoverTotalFreq*)cover)->intersect(attr[l], query->weights);
        Supports a = copySupports(cover->sup_class);
        Support b = cover->support;
        cover->backtrack();*/
        cover->intersect(attr[l], query->weights);
        /*if (a[0] != cover->sup_class[0] || a[1] != cover->sup_class[1]){
            cout << "pb intersect sc" << endl;
        }
        else cout << "ok intersect sc" << endl;
        if (b != cover->support) {
            cout << "pb intersect sup" << endl;
        }
        else cout << "ok intersect sup" << endl;
        deleteSupports(a);*/

//        Supports c = ((RCoverTotalFreq*)cover)->getSupportPerClass(query->weights);
//        Support d = cover->support;
        sups_sc[l][l] = cover->getSupportPerClass(query->weights);
        /*if (c[0] != sups_sc[l][l][0] || c[1] != sups_sc[l][l][1]){
            cout << "pb getsc sc" << endl;
        }
        else cout << "ok getsc sc" << endl;
        if (d != cover->support) {
            cout << "pb getsc sup" << endl;
        }
        else cout << "ok getsc sup" << endl;*/
//        deleteSupports(c);


        sups[l][l] = cover->getSupport();
        for (int i = l + 1; i < attr.size(); ++i) {
//            cout << "j = " << i << endl;
            //pair<Supports, Support> e = ((RCoverTotalFreq*)cover)->temporaryIntersect(attr[i], query->weights);
            pair<Supports, Support> p = cover->temporaryIntersect(attr[i], query->weights);
            //cout << e.first[0] << ":" << e.first[1] << " - " << p.first[0] << ":" << p.first[1] << endl;
            /*if (e.first[0] != p.first[0] || e.first[1] != p.first[1]){
                cout << "pb tempint sc" << endl;
            }
            else cout << "ok tempint sc" << endl;
            if (e.second != p.second) {
                cout << "pb tempint sup" << endl;
            }
            else cout << "ok tempint sup" << endl;*/
            //deleteSupports(e.first);
            sups_sc[l][i] = p.first;
            sups[l][i] = p.second;
        }
        cover->backtrack();
    }
    auto stop_comp = high_resolution_clock::now();
    comptime += duration_cast<milliseconds>(stop_comp - start_comp).count() / 1000.0;

    TreeTwo best_tree;

    // find the best tree for each feature
    for (int i = 0; i < attr.size(); ++i) {
        if (verbose) cout << "root test: " << attr[i] << endl;

        TreeTwo feat_best_tree;
        feat_best_tree.root_data->test = attr[i];

        Supports idsc = sups_sc[i][i];
        Support ids = sups[i][i];
//        Support ids = sumSupports(idsc);
        Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc);
        Support igs = root_sup - ids;

        //feature to left
        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup) {
            if (verbose) cout << "root impossible de splitter...on backtrack" << endl;
            continue;
        }

        feat_best_tree.root_data->left = feat_best_tree.left_data;
        feat_best_tree.root_data->right = feat_best_tree.right_data;

        // the feature at root cannot be splitted at left. It is then a leaf node
        if (igs < 2 * query->minsup) {
            ErrorValues ev = query->computeErrorValues(igsc);
            feat_best_tree.left_data->error = ev.error;
            feat_best_tree.left_data->test = ev.maxclass;
            if (verbose)
                cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = "
                     << feat_best_tree.left_data->error << " on backtrack" << endl;
        }
            // the root node can theorically be split at left
        else {
            if (verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            ErrorValues ev = query->computeErrorValues(igsc);
            feat_best_tree.left_data->error = min(ev.error, best_tree.root_data->error);
            feat_best_tree.left_data->leafError = ev.error;
            feat_best_tree.left_data->test = ev.maxclass;

            if (!floatEqual(ev.error, lb)) {
                Error tmp = feat_best_tree.left_data->error;
                for (int j = 0; j < attr.size(); ++j) {
                    if (verbose) cout << "left test: " << attr[j] << endl;
                    if (attr[i] == attr[j]) {
                        if (verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }
                    Supports jdsc = sups_sc[j][j], idjdsc = sups_sc[min(i, j)][max(i, j)], igjdsc = newSupports();
                    subSupports(jdsc, idjdsc, igjdsc);
                    Support jds = sups[j][j];
//                    Support jds = sumSupports(jdsc);
                    Support idjds = sups[min(i, j)][max(i, j)];
//                    Support idjds = sumSupports(idjdsc);
                    Support igjds = jds - idjds;
//                    Support igjds =  sumSupports(igjdsc);
                    Support igjgs = igs - igjds;

                    // the root node can in practice be split into two children
                    if (igjgs >= query->minsup && igjds >= query->minsup) {
                        if (verbose) cout << "le left testé peut splitter. on le regarde" << endl;

                        ErrorValues ev2 = query->computeErrorValues(igjdsc);
                        if (verbose) cout << "le left a droite produit une erreur de " << ev2.error << endl;

                        if (ev2.error >= min(best_tree.root_data->error, feat_best_tree.left_data->error)) {
                            if (verbose)
                                cout << "l'erreur gauche du left montre rien de bon. best root: " << best_tree.root_data->error
                                     << " best left: " << feat_best_tree.left_data->error << " Un autre left..." << endl;
                            continue;
                        }

                        Supports igjgsc = newSupports();
                        subSupports(igsc, igjdsc, igjgsc);
                        ErrorValues ev1 = query->computeErrorValues(igjgsc);
                        if (verbose) cout << "le left a gauche produit une erreur de " << ev1.error << endl;

                        if (ev1.error + ev2.error < min(best_tree.root_data->error, feat_best_tree.left_data->error)) {
                            feat_best_tree.left_data->error = ev1.error + ev2.error;
                            if (verbose)
                                cout << "ce left ci donne une meilleure erreur que les précédents left: "
                                     << feat_best_tree.left_data->error << endl;
                            feat_best_tree.left1_data->error = ev1.error;
                            feat_best_tree.left1_data->test = ev1.maxclass;
                            feat_best_tree.left2_data->error = ev2.error;
                            feat_best_tree.left2_data->test = ev2.maxclass;
                            feat_best_tree.left_data->test = attr[j];
                            feat_best_tree.left_data->left = feat_best_tree.left1_data;
                            feat_best_tree.left_data->right = feat_best_tree.left2_data;
                            feat_best_tree.left_data->size = 3;

                            if (floatEqual(feat_best_tree.left_data->error, lb)) break;
                        } else {
                            if (verbose)
                                cout << "l'erreur du left = " << ev1.error + ev2.error
                                     << " n'ameliore pas l'existant. Un autre left..." << endl;
                        }
                        deleteSupports(igjgsc);
                    } else if (verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                    deleteSupports(igjdsc);
                }
                if (floatEqual(feat_best_tree.left_data->error, tmp) && verbose)
                    cout << "aucun left n'a su splitter. on garde le root gauche comme leaf avec erreur: "
                         << feat_best_tree.left_data->error << endl;
            } else {
                if (verbose)
                    cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: "
                         << feat_best_tree.left_data->error << endl;
            }
        }


        //feature to right
        if (feat_best_tree.left_data->error < best_tree.root_data->error) {
            if (verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

            // the feature at root cannot be split at right. It is then a leaf node
            if (ids < 2 * query->minsup) {
                ErrorValues ev = query->computeErrorValues(idsc);
                feat_best_tree.right_data->error = ev.error;
                feat_best_tree.right_data->test = ev.maxclass;
                if (verbose)
                    cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = "
                         << feat_best_tree.right_data->error << " on backtrack" << endl;
            } else {
                if (verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
                // at worst it can't in practice and error will be considered as leaf node
                // so the error is initialized at this case
                ErrorValues ev = query->computeErrorValues(idsc);
                Error remainingError = best_tree.root_data->error - feat_best_tree.left_data->error;
                feat_best_tree.right_data->error = min(ev.error, remainingError);
                feat_best_tree.right_data->leafError = ev.error;
                feat_best_tree.right_data->test = ev.maxclass;

                Error tmp = feat_best_tree.right_data->error;

                if (!floatEqual(ev.error, lb)) {
                    for (int j = 0; j < attr.size(); ++j) {
                        if (verbose) cout << "right test: " << attr[j] << endl;
                        if (attr[i] == attr[j]) {
                            if (verbose)
                                cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                            continue;
                        }

                        Supports idjdsc = sups_sc[min(i, j)][max(i, j)], idjgsc = newSupports();
                        subSupports(idsc, idjdsc, idjgsc);
                        Support idjds = sups[min(i, j)][max(i, j)];
//                        Support idjds = sumSupports(idjdsc);
                        Support idjgs = ids - idjds;
//                        Support idjgs = sumSupports(idjgsc);

                        // the root node can in practice be split into two children
                        if (idjgs >= query->minsup && idjds >= query->minsup) {
                            if (verbose) cout << "le right testé peut splitter. on le regarde" << endl;
                            ErrorValues ev1 = query->computeErrorValues(idjgsc);
                            if (verbose) cout << "le right a gauche produit une erreur de " << ev1.error << endl;

                            if (ev1.error >= min(remainingError, feat_best_tree.right_data->error)) {
                                if (verbose)
                                    cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                continue;
                            }

                            ErrorValues ev2 = query->computeErrorValues(idjdsc);
                            if (verbose) cout << "le right a droite produit une erreur de " << ev2.error << endl;
                            if (ev1.error + ev2.error < min(remainingError, feat_best_tree.right_data->error)) {
                                feat_best_tree.right_data->error = ev1.error + ev2.error;
                                if (verbose)
                                    cout << "ce right ci donne une meilleure erreur que les précédents right: "
                                         << feat_best_tree.right_data->error << endl;
                                feat_best_tree.right1_data->error = ev1.error;
                                feat_best_tree.right1_data->test = ev1.maxclass;
                                feat_best_tree.right2_data->error = ev2.error;
                                feat_best_tree.right2_data->test = ev2.maxclass;
                                feat_best_tree.right_data->test = attr[j];
                                feat_best_tree.right_data->left = feat_best_tree.right1_data;
                                feat_best_tree.right_data->right = feat_best_tree.right2_data;
                                feat_best_tree.right_data->size = 3;

                                if (floatEqual(feat_best_tree.right_data->error, lb)) break;
                            } else {
                                if (verbose)
                                    cout << "l'erreur du right = " << ev1.error + ev2.error
                                         << " n'ameliore pas l'existant. Un autre right..." << endl;
                            }
                        } else if (verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                        deleteSupports(idjgsc);
                    }
                    if (floatEqual(feat_best_tree.right_data->error, tmp))
                        if (verbose)
                            cout << "aucun right n'a su splitter. on garde le root droite comme leaf avec erreur: "
                                 << feat_best_tree.right_data->error << endl;
                } else if (verbose)
                    cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: "
                         << feat_best_tree.right_data->error << endl;
            }

            if (feat_best_tree.left_data->error + feat_best_tree.right_data->error < best_tree.root_data->error) {
//                cout << "o1" << endl;
                feat_best_tree.root_data->error = feat_best_tree.left_data->error + feat_best_tree.right_data->error;
                feat_best_tree.root_data->size = feat_best_tree.left_data->size + feat_best_tree.right_data->size;

                if (verbose)
                    cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: "
                         << best_tree.root_data->error << endl;
                best_tree = feat_best_tree;
            } else {
                if (verbose) cout << "cet arbre n'est pas mieux que le meilleur jusque là." << endl;
            }
        }
        deleteSupports(igsc);
    }
    for (int k = 0; k < attr.size(); ++k) {
        for (int i = k; i < attr.size(); ++i) {
            deleteSupports(sups_sc[k][i]);
        }
        delete [] sups_sc[k];
        delete [] sups[k];
    }
    delete [] sups_sc;
    delete [] sups;
    if (verbose) cout << "root: " << best_tree.root_data->test << " left: " << best_tree.left_data->test << " right: " << best_tree.right_data->test << endl;
    if (verbose)
        cout << "le1: " << best_tree.left1_data->error << " le2: " << best_tree.left2_data->error << " re1: " << best_tree.right1_data->error << " re2: "
             << best_tree.right2_data->error << endl;
    if (verbose)
        cout << "ble: " << best_tree.left_data->error << " bre: " << best_tree.right_data->error << " broe: " << best_tree.root_data->error << endl;
    if (verbose)
        cout << "lc1: " << best_tree.left1_data->test << " lc2: " << best_tree.left2_data->test << " rc1: " << best_tree.right1_data->test << " rc2: "
             << best_tree.right2_data->test << endl;
    if (verbose) cout << "blc: " << best_tree.left_data->test << " brc: " << best_tree.right_data->test << endl;
//    cout << "temps find: " << (clock() - tt) / (float) CLOCKS_PER_SEC << " ";

    if (best_tree.root_data->test != -1) {

        node->data = (QueryData *) best_tree.root_data;
        setItem((QueryData_Best *) node->data, itemset, trie);
//        best_tree.root_data->size += best_tree.root_data->left->size + best_tree.root_data->right->size;

//        cout << best_tree.root_data->test << " " << best_tree.root_data->size << endl;

        auto stop = high_resolution_clock::now();
        spectime += duration_cast<milliseconds>(stop - start).count() / 1000.0;

        return node;
    } else {
        //error not lower than ub
//            cout << "cale" << endl;
        ErrorValues ev = query->computeErrorValues(cover);
        node->data = (QueryData *) new QueryData_Best();
        ((QueryData_Best *) node->data)->error = FLT_MAX;
        ((QueryData_Best *) node->data)->leafError = ev.error;
        ((QueryData_Best *) node->data)->test = ev.maxclass;
//            cout << "cc2" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        auto stop = high_resolution_clock::now();
        spectime += duration_cast<milliseconds>(stop - start).count() / 1000.0;
        return node;
    }

}