//
// Created by Gael Aglin on 26/09/2020.
//

#include "depthTwoComputer.h"

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
    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    Support root_sup = cover->getSupport();

    // update the next candidates list by removing the one already added
    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for(auto& attribute : attributes_to_visit) {
        if (last_added == attribute) continue;
        attr.push_back(attribute);
    }

    // compute the different support per class we need to perform the search
    // only a few mandatory are computed. The remaining are derived from them
    auto start_comp = high_resolution_clock::now();
    Supports **sups_sc = new Supports *[attr.size()];
    Support **sups = new Support* [attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        sups_sc[l] = new Supports[attr.size()];
        sups[l] = new Support[attr.size()];
        cover->intersect(attr[l]);
        sups_sc[l][l] = cover->getSupportPerClass();
        sups[l][l] = cover->getSupport();
        for (int i = l + 1; i < attr.size(); ++i) {
            pair<Supports, Support> p = cover->temporaryIntersect(attr[i]);
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



/*TrieNode* getdepthtwotrees(RCover* cover,
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
    Supports root_sup_clas = copySupports(cover->getSupportPerClass());
    Support root_sup = cover->getSupport();

    // update the next candidates list by removing the one already added
    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size - 1);
    for (int m = 0; m < attributes_to_visit.size; ++m) {
        if (last_added == attributes_to_visit[m]) continue;
        attr.push_back(attributes_to_visit[m]);
    }

    // compute the different support per class we need to perform the search
    // only a few mandatory are computed. The remaining are derived from them
    auto start_comp = high_resolution_clock::now();
    Supports **sups = new Supports *[attr.size()];
    for (int l = 0; l < attr.size(); ++l) {
        sups[l] = new Supports[attr.size()];
        cover->intersect(attr[l]);
        sups[l][l] = cover->getSupportPerClass();
        for (int i = l + 1; i < attr.size(); ++i) sups[l][i] = cover->intersectAndClass(attr[i]);
        cover->backtrack();
    }
    auto stop_comp = high_resolution_clock::now();
    comptime += duration_cast<milliseconds>(stop_comp - start_comp).count() / 1000.0;
//    cout << " temps comp: " << (clock() - ttt) / (float) CLOCKS_PER_SEC << " ";
//    exit(0);


    Attribute root = -1, left = -1, right = -1;
    Error best_root_error = ub,
            best_left_error1 = FLT_MAX,
            best_left_error2 = FLT_MAX,
            best_right_error1 = FLT_MAX,
            best_right_error2 = FLT_MAX,
            best_left_error = FLT_MAX,
            best_right_error = FLT_MAX;

    QueryData_Best* root_node = new QueryData_Best();
    QueryData_Best* left_node = new QueryData_Best();
    QueryData_Best* right_node = new QueryData_Best();
    QueryData_Best* left1_node = new QueryData_Best();
    QueryData_Best* left2_node = new QueryData_Best();
    QueryData_Best* right1_node = new QueryData_Best();
    QueryData_Best* right2_node = new QueryData_Best();

    Error root_leaf_error = query->computeErrorValues(cover).error,
            best_left_leafError = FLT_MAX,
            best_right_leafError = FLT_MAX;

    Class best_left_class1 = -1,
            best_left_class2 = -1,
            best_right_class1 = -1,
            best_right_class2 = -1,
            best_left_class = -1,
            best_right_class = -1;


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
                        if (verbose) cout << "le left a droite produit une erreur de " << tmp_left_error2 << endl;

                        if (tmp_left_error2 >= min(best_root_error, best_feat_left_error)) {
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
                        if (verbose) cout << "le left a gauche produit une erreur de " << tmp_left_error1 << endl;

                        if (tmp_left_error1 + tmp_left_error2 < min(best_root_error, best_feat_left_error)) {
                            best_feat_left_error = tmp_left_error1 + tmp_left_error2;
                            if (verbose)
                                cout << "ce left ci donne une meilleure erreur que les précédents left: "
                                     << best_feat_left_error << endl;
                            best_feat_left_error1 = tmp_left_error1;
                            best_feat_left_error2 = tmp_left_error2;
                            best_feat_left_class1 = tmp_left_class1;
                            best_feat_left_class2 = tmp_left_class2;
                            feat_left = attr[j];
                            if (floatEqual(best_feat_left_error, lb)) break;
                        } else {
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
                            if (verbose) cout << "le right a gauche produit une erreur de " << tmp_right_error1 << endl;

                            if (tmp_right_error1 >=
                                min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                if (verbose)
                                    cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                                continue;
                            }

                            ev = query->computeErrorValues(idjdsc);
                            Error tmp_right_error2 = ev.error;
                            Class tmp_right_class2 = ev.maxclass;
                            if (verbose) cout << "le right a droite produit une erreur de " << tmp_right_error2 << endl;
                            if (tmp_right_error1 + tmp_right_error2 <
                                min((best_root_error - best_feat_left_error), best_feat_right_error)) {
                                best_feat_right_error = tmp_right_error1 + tmp_right_error2;
                                if (verbose)
                                    cout << "ce right ci donne une meilleure erreur que les précédents right: "
                                         << best_feat_right_error << endl;
                                best_feat_right_error1 = tmp_right_error1;
                                best_feat_right_error2 = tmp_right_error2;
                                best_feat_right_class1 = tmp_right_class1;
                                best_feat_right_class2 = tmp_right_class2;
                                feat_right = attr[j];
                                if (floatEqual(best_feat_right_error, lb)) break;
                            } else {
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
            } else {
//                cout << "o2" << endl;
//                cout << "feat_left = " << feat_left << " and feat_right = " << feat_right << endl;
//                cout << best_left_corrects << endl;
//                if (best_left_corrects) deleteSupports(best_left_corrects);
//                if (best_left_falses) deleteSupports(best_left_falses);
//                if (best_right_corrects) deleteSupports(best_right_corrects);
//                if (best_right_falses) deleteSupports(best_right_falses);
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

//            cout << "cc1" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
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
        ((QueryData_Best *) node->data)->size = 1;
        ((QueryData_Best *) node->data)->left = nullptr;
        ((QueryData_Best *) node->data)->right = nullptr;
//            cout << "cc2" << endl;
//        cout << " temps total: " << (clock() - tt) / (float) CLOCKS_PER_SEC << endl;
        auto stop = high_resolution_clock::now();
        spectime += duration_cast<milliseconds>(stop - start).count() / 1000.0;
        return node;
    }

}*/
