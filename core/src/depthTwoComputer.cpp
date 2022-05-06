//
// Created by Gael Aglin on 26/09/2020.
//

#include "depthTwoComputer.h"
#include "rCoverTotalFreq.h"

struct SupportsDeleter {
    void operator()(const Supports p) {
        delete[] p;
    }
};

void setItem(QueryData_Best* node_data, Array<Item> itemset, Trie* trie){
    if (node_data->left){
        Array<Item> itemset_left = addItem(itemset, item(node_data->left->test, 0));
        TrieNode *node_left = trie->insert(itemset_left);
        node_left->data = (QueryData *) node_data->left;
        setItem((QueryData_Best *)node_left->data, itemset_left, trie);
        itemset_left.free();
    }

    if (node_data->right){
        Array<Item> itemset_right = addItem(itemset, item(node_data->right->test, 1));
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
        if (verbose) cout << "infeasible case. ub = " << ub << " lb = " << lb << endl;
        return node;
    }

    // The fact to not bound the search make it find the best solution in any case and remove the chance to recall this
    // function for the same node with an higher upper bound. Since this function is not exponential, we can afford that.
    ub = FLT_MAX;

    //local variable to make the function verbose or not. Can be improved :-)
    bool local_verbose = verbose;

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
        cover->intersect(attr[l]);
        sups_sc[l][l] = copySupports(cover->getSupportPerClass());
        sups[l][l] = cover->getSupport();

        // compute value for second level
        for (int i = l + 1; i < attr.size(); ++i) {

            pair<Supports, Support> p = cover->temporaryIntersect(attr[i]);
            sups_sc[l][i] = p.first;
            sups[l][i] = p.second;

            // in case of timeout: no more time
            if (query->timeLimit > 0 and duration<float>(high_resolution_clock::now() - query->startTime).count() >= (float)query->timeLimit) {
                for (int k = 0; k <= l; ++k) {
                    for (int j = k; j <= i; ++j) {
                        deleteSupports(sups_sc[k][j]);
                    }
                    delete [] sups_sc[k];
                    delete [] sups[k];
                }
                delete [] sups_sc;
                delete [] sups;
                deleteSupports(root_sup_clas);
                cover->backtrack();
                node->data = query->initData(cover); // no need to update the error
                return node;
            }

            
        }
        // backtrack to recover the cover state
        cover->backtrack();
    }


    auto* best_tree = new TreeTwo();

    // find the best tree for each feature
    for (int i = 0; i < attr.size(); ++i) {

        // in case of timeout: no more time
        if (query->timeLimit > 0 and duration<float>(high_resolution_clock::now() - query->startTime).count() >= (float)query->timeLimit) break;

        if (local_verbose) cout << "root test: " << attr[i] << endl;
        //cout << "beeest " << best_tree->root_data->error << endl;

        // best tree for the current feature
        auto* feat_best_tree = new TreeTwo();
        // set the root to the current feature
        feat_best_tree->root_data->test = attr[i];
        // compute its error and set it as initial error
        LeafInfo ev = query->computeLeafInfo(cover);
        feat_best_tree->root_data->leafError = ev.error;
        // feat_best_tree.root_data->test = ev.maxclass;

        Supports idsc = sups_sc[i][i];
        Support ids = sups[i][i]; // Support ids = sumSupports(idsc);
        unique_ptr<SupportClass, SupportsDeleter> igsc(newSupports());
        // Supports igsc = newSupports();
        subSupports(root_sup_clas, idsc, igsc.get());
        Support igs = root_sup - ids;

        
        
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //          LEFT CHILD       //
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//

        // the feature cannot be root since its two children will not fullfill the minsup constraint
        if (igs < query->minsup || ids < query->minsup) {
            if (local_verbose) cout << "root impossible de splitter...on backtrack" << endl;
            delete feat_best_tree;
            // deleteSupports(igsc);
            continue;
        }

        feat_best_tree->root_data->left = new QueryData_Best();

        // the feature at root cannot be splitted at left. It is then a leaf node
        if (igs < 2 * query->minsup) {
            LeafInfo ev = query->computeLeafInfo(igsc.get());
            feat_best_tree->root_data->left->error = ev.error;
            feat_best_tree->root_data->left->test = ev.maxclass;
            if (local_verbose)
                cout << "root gauche ne peut théoriquement spliter; donc feuille. erreur gauche = " << feat_best_tree->root_data->left->error << " on backtrack" << endl;
        }
        // the root node can theorically be split at left
        else {
            if (local_verbose) cout << "root gauche peut théoriquement spliter. Creusons plus..." << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            LeafInfo ev = query->computeLeafInfo(igsc.get());
            Error feat_ub = best_tree->root_data->error;
            feat_best_tree->root_data->left->leafError = ev.error;
            feat_best_tree->root_data->left->test = ev.maxclass;

            // no need to search deep in the tree if minimal error is reached
            if (floatEqual(ev.error, 0)) {
                feat_best_tree->root_data->left->error = ev.error;
                if (local_verbose) cout << "l'erreur du root gauche est minimale. on garde le root gauche comme leaf avec erreur: " << feat_best_tree->root_data->left->error << endl;
            }
            else {
                for (int j = 0; j < attr.size(); ++j) {

                    // in case of timeout: no more time
                    if (query->timeLimit > 0 and duration<float>(high_resolution_clock::now() - query->startTime).count() >= (float)query->timeLimit) {
                        if (feat_best_tree->root_data->left->error > ev.error) {
                            feat_best_tree->root_data->left->error = ev.error;
                            feat_best_tree->root_data->left->test = ev.maxclass;
                        }
                        break;
                    }

                    if (local_verbose) cout << "left test: " << attr[j] << endl;
                    if (attr[i] == attr[j]) {
                        if (local_verbose) cout << "left pareil que le parent ou non sup...on essaie un autre left" << endl;
                        continue;
                    }
                    Supports jdsc = sups_sc[j][j], idjdsc = sups_sc[min(i, j)][max(i, j)];//, igjdsc = newSupports();
                    unique_ptr<SupportClass, SupportsDeleter> igjdsc(newSupports());
                    subSupports(jdsc, idjdsc, igjdsc.get());
                    Support jds = sups[j][j]; // Support jds = sumSupports(jdsc);
                    Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                    Support igjds = jds - idjds; // Support igjds =  sumSupports(igjdsc);
                    Support igjgs = igs - igjds;

                    // the left node cannot in practice be split into two children
                    if (igjgs < query->minsup || igjds < query->minsup) {
                        if (local_verbose) cout << "le left testé ne peut splitter en pratique...un autre left!!!" << endl;
                        // deleteSupports(igjdsc);
                        continue;
                    }

                    if (local_verbose) cout << "le left testé peut splitter. on le regarde" << endl;

                    LeafInfo ev2 = query->computeLeafInfo(igjdsc.get());
                    if (local_verbose) cout << "le left a droite produit une erreur de " << ev2.error << endl;

                    if (ev2.error >= feat_ub) {
                        if (local_verbose) cout << "l'erreur gauche du left montre rien de bon. best root: " << best_tree->root_data->error << " best left: " << feat_best_tree->root_data->left->error << " Un autre left..." << endl;
                        // deleteSupports(igjdsc);
                        continue;
                    }

                    // Supports igjgsc = newSupports();
                    unique_ptr<SupportClass, SupportsDeleter> igjgsc(newSupports());
                    subSupports(igsc.get(), igjdsc.get(), igjgsc.get());
                    LeafInfo ev1 = query->computeLeafInfo(igjgsc.get());
                    if (local_verbose) cout << "le left a gauche produit une erreur de " << ev1.error << endl;

                    // error worse than existing tree error
                    if (ev1.error + ev2.error >= feat_ub) {
                        if (local_verbose) cout << "l'erreur du left = " << ev1.error + ev2.error << " n'ameliore pas l'existant. Un autre left..." << endl;
                        // deleteSupports(igjgsc);
                        // deleteSupports(igjdsc);
                        continue;
                    }

                    // in case error found is equal to leaf error, we prefer a shallow tree
                    if ( floatEqual(ev1.error + ev2.error, ev.error) ) {
                        feat_best_tree->root_data->left->error = ev.error;
                    }
                    else { 
                        feat_best_tree->root_data->left->error = ev1.error + ev2.error;
                        if (local_verbose) cout << "ce left ci donne une meilleure erreur que les précédents left: " << feat_best_tree->root_data->left->error << endl;
                        
                        if (!feat_best_tree->root_data->left->left){
                            feat_best_tree->root_data->left->left = new QueryData_Best();
                            feat_best_tree->root_data->left->right = new QueryData_Best();
                        }

                        feat_best_tree->root_data->left->left->error = ev1.error;
                        feat_best_tree->root_data->left->left->test = ev1.maxclass;
                        feat_best_tree->root_data->left->right->error = ev2.error;
                        feat_best_tree->root_data->left->right->test = ev2.maxclass;
                        feat_best_tree->root_data->left->test = attr[j];
                        feat_best_tree->root_data->left->size = 3;

                    }
                    feat_ub = ev1.error + ev2.error;
                    // deleteSupports(igjgsc);
                    // deleteSupports(igjdsc);
                    if (floatEqual(ev1.error + ev2.error, 0)) break;
                }

                // there is no left child coupled to the root to produce lower error than the best tree so far. No need to look at right
                // the second case can happen when timeout is reached
                if (floatEqual(feat_best_tree->root_data->left->error, FLT_MAX) 
                        || feat_best_tree->root_data->left->error >= best_tree->root_data->error // in case of timeout, this can happen
                    ){
                    if (local_verbose) cout << "aucun left n'a su améliorer l'arbre existant: " << feat_best_tree->root_data->left->error << " on garde l'ancien arbre" << endl;
                    delete feat_best_tree;
                    // deleteSupports(igsc);
                    continue; // test new root
                }
            }
            // deleteSupports(igsc);
        }


        
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //         RIGHT CHILD       //
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%//

        if (local_verbose) cout << "vu l'erreur du root gauche et du left. on peut tenter quelque chose à droite" << endl;

        feat_best_tree->root_data->right = new QueryData_Best();

        // the feature at root cannot be split at right. It is then a leaf node
        if (ids < 2 * query->minsup) {
            LeafInfo ev = query->computeLeafInfo(idsc);
            feat_best_tree->root_data->right->error = ev.error;
            feat_best_tree->root_data->right->test = ev.maxclass;
            if (local_verbose) cout << "root droite ne peut théoriquement spliter; donc feuille. erreur droite = " << feat_best_tree->root_data->right->error << " on backtrack" << endl;
        } 
        else {
            if (local_verbose) cout << "root droite peut théoriquement spliter. Creusons plus..." << endl;
            // at worst it can't in practice and error will be considered as leaf node
            // so the error is initialized at this case
            LeafInfo ev = query->computeLeafInfo(idsc);
            Error feat_ub = best_tree->root_data->error - feat_best_tree->root_data->left->error;
            feat_best_tree->root_data->right->leafError = ev.error;
            feat_best_tree->root_data->right->test = ev.maxclass;

            if (floatEqual(feat_best_tree->root_data->left->error + ev.error, lb)) {
                if (local_verbose) cout << "l'erreur du root droite est minimale. on garde le root droite comme leaf avec erreur: " << feat_best_tree->root_data->right->error << endl;
                feat_best_tree->root_data->right->error = ev.error;
                feat_best_tree->root_data->error = feat_best_tree->root_data->left->error + feat_best_tree->root_data->right->error;
                best_tree->replaceTree(feat_best_tree);
                // best_tree = move(feat_best_tree);
                break; // best is found
            }
            
            for (int j = 0; j < attr.size(); ++j) {

                // in case of timeout: no more time
                if (query->timeLimit > 0 and duration<float>(high_resolution_clock::now() - query->startTime).count() >= (float)query->timeLimit) {
                    if (feat_best_tree->root_data->right->error > ev.error) {
                        feat_best_tree->root_data->right->error = ev.error;
                        feat_best_tree->root_data->right->test = ev.maxclass;
                    }
                    break;
                }

                if (local_verbose) cout << "right test: " << attr[j] << endl;
                if (attr[i] == attr[j]) {
                    if (local_verbose)
                        cout << "right pareil que le parent ou non sup...on essaie un autre right" << endl;
                    continue;
                }

                Supports idjdsc = sups_sc[min(i, j)][max(i, j)];//, idjgsc = newSupports();
                unique_ptr<SupportClass, SupportsDeleter> idjgsc(newSupports());
                subSupports(idsc, idjdsc, idjgsc.get());
                Support idjds = sups[min(i, j)][max(i, j)]; // Support idjds = sumSupports(idjdsc);
                Support idjgs = ids - idjds; // Support idjgs = sumSupports(idjgsc);

                // the root node can in practice be split into two children
                if (idjgs < query->minsup || idjds < query->minsup) {
                    if (local_verbose) cout << "le right testé ne peut splitter...un autre right!!!" << endl;
                    continue; // test next right
                }
                
                LeafInfo ev1 = query->computeLeafInfo(idjgsc.get());
                if (local_verbose) cout << "le right a gauche produit une erreur de " << ev1.error << endl;

                if (ev1.error >= feat_ub) {
                    if (local_verbose) cout << "l'erreur gauche du right montre rien de bon. Un autre right..." << endl;
                    // deleteSupports(idjgsc);
                    continue;
                }

                LeafInfo ev2 = query->computeLeafInfo(idjdsc);
                if (local_verbose) cout << "le right a droite produit une erreur de " << ev2.error << endl;
                
                if (ev1.error + ev2.error >= feat_ub) {
                    if (local_verbose) cout << "l'erreur du right = " << ev1.error + ev2.error << " n'ameliore pas l'existant. Un autre right..." << endl;
                    // deleteSupports(idjgsc);
                    continue; // test next right
                }

                // in case error found is equal to leaf error, we prefer a shallow tree
                if ( floatEqual(ev1.error + ev2.error, ev.error) ) {
                    feat_best_tree->root_data->right->error = ev.error;
                }
                else {
                    feat_best_tree->root_data->right->error = ev1.error + ev2.error;
                    if (local_verbose) cout << "ce right ci donne une meilleure erreur que les précédents right: " << feat_best_tree->root_data->right->error << endl;
                    
                    if (!feat_best_tree->root_data->right->left){
                        feat_best_tree->root_data->right->left = new QueryData_Best();
                        feat_best_tree->root_data->right->right = new QueryData_Best();
                    }

                    feat_best_tree->root_data->right->left->error = ev1.error;
                    feat_best_tree->root_data->right->left->test = ev1.maxclass;
                    feat_best_tree->root_data->right->right->error = ev2.error;
                    feat_best_tree->root_data->right->right->test = ev2.maxclass;
                    feat_best_tree->root_data->right->test = attr[j];
                    feat_best_tree->root_data->right->size = 3;
                }
                feat_ub = ev1.error + ev2.error;
                // deleteSupports(idjgsc);
                if (floatEqual(feat_best_tree->root_data->right->error + feat_best_tree->root_data->left->error, lb) or floatEqual(ev1.error + ev2.error, 0)) break;
            }

            // there is no right child coupled to the root and left to produce lower error than the best tree so far.
            if (floatEqual(feat_best_tree->root_data->right->error, FLT_MAX)  
                    || feat_best_tree->root_data->right->error >= best_tree->root_data->error - feat_best_tree->root_data->left->error // in case there is timeout, this can happen
                ){
                if (local_verbose) cout << "pas d'arbre mieux que le meilleur jusque là." << endl;
                delete feat_best_tree;
                continue; // test new root
            }

            feat_best_tree->root_data->error = feat_best_tree->root_data->left->error + feat_best_tree->root_data->right->error;
            feat_best_tree->root_data->size += feat_best_tree->root_data->left->size + feat_best_tree->root_data->right->size;

            best_tree->replaceTree(feat_best_tree);
            if (local_verbose) cout << "ce triple (root, left, right) ci donne une meilleure erreur que les précédents triplets: " << best_tree->root_data->error << " " << best_tree->root_data->test << endl;
            
            if (floatEqual(best_tree->root_data->error, lb)) {
                if (local_verbose) cout << "The best tree is found" << endl;
                break;
            }
        }

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
    deleteSupports(root_sup_clas);
    if (local_verbose && best_tree->root_data && best_tree->root_data->left && best_tree->root_data->right) cout << "root: " << best_tree->root_data->test << " left: " << best_tree->root_data->left->test << " right: " << best_tree->root_data->right->test << endl;

    if (best_tree->root_data->test != -1) {
        if (best_tree->root_data->size == 3 && best_tree->root_data->left->test == best_tree->root_data->right->test && floatEqual(best_tree->root_data->leafError, best_tree->root_data->left->error + best_tree->root_data->right->error)) {
            best_tree->root_data->size = 1;
            best_tree->root_data->error = best_tree->root_data->leafError;
            best_tree->root_data->test = best_tree->root_data->right->test;
            delete best_tree->root_data->left;
            best_tree->root_data->left = nullptr;
            delete best_tree->root_data->right;
            best_tree->root_data->right = nullptr;
            node->data = (QueryData *)best_tree->root_data;
            if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->error) << endl;
            return node;
        }

        node->data = (QueryData *) best_tree->root_data;
        setItem((QueryData_Best *) node->data, itemset, trie);

        if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->error) << endl;
        return node;
    } else {
        // cout << "this case " << endl;
        //no tree with error lower than considering the node as a leaf
        delete best_tree;
        node->data = query->initData(cover);
        ((QueryData_Best *) node->data)->error = ((QueryData_Best *) node->data)->leafError;
        if (verbose) cout << "best twotree error = " << to_string(((QueryData_Best *) node->data)->error) << endl;
        return node;
        // LeafInfo ev = query->computeLeafInfo(cover);
        // delete best_tree;
        // node->data = (QueryData *) new QueryData_Best();
        // ((QueryData_Best *) node->data)->error = ev.error;
        // ((QueryData_Best *) node->data)->leafError = ev.error;
        // ((QueryData_Best *) node->data)->test = ev.maxclass;
        // if (verbose) cout << "best twotree error = " << to_string(best_tree->root_data->error) << endl;
        // return node;
    }

}