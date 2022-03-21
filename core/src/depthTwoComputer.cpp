//
// Created by Gael Aglin on 26/09/2020.
//

#include "depthTwoComputer.h"
#include "search_base.h"
#include "cache_trie.h"
//#include "search_nocache.h"

struct ErrorvalsDeleter {
    void operator()(const ErrorVals p) {
        delete[] p;
    }
};


//void addTreeToCache(Node* node, const Itemset &itemset, Cache* cache){

void addTreeToCache(DepthTwo_NodeData* node_data,  const Itemset &itemset, Cache* cache){
//    printItemset(itemset, true);
//    auto* node_data = node->data;
//    if(itemset.size() == 5 and itemset.at(0) == 0 and itemset.at(1) == 2 and itemset.at(2) == 11 and itemset.at(3) == 12 and itemset.at(4) == 28) {
//        cout << "coucouAAAAAAA " << node->data->error << " " << node->data->test << endl;// << " " << node->data->left->data->error << " " << node->data->right->data->error << endl;
//        if (node->data->test == 8) exit(0);
//    }
    if (node_data->test >= 0) { // not a leaf

//        if(itemset.size() == 3 and itemset.at(0) == 16 and itemset.at(1) == 38 and itemset.at(2) == 40) {
//            cout << "changed " << node << endl;
//            cout << "data before " << node_data << endl;
//        }

        Itemset itemset_left = addItem(itemset, item(node_data->test, NEG_ITEM));
        pair<Node *, bool> res_left = cache->insert(itemset_left);
        Node *node_left = res_left.first;
        if (res_left.second) { //new node. data is still null
            node_left->data = new TrieNodeData(*(((DepthTwo_NodeData*)node_data)->left));
//            node_left->data = new TrieNodeData();
//            *((TrieNodeData*)node_left->data) = *(node_data->left);
        }
        else *((TrieNodeData*)node_left->data) = *(node_data->left);
//        node_data->left = node_left;
//        ((TrieNode*)node_left)->search_parents.push_back((TrieNode*)node);
//        ((TrieNode*)node_left)->search_parents.insert(make_pair((TrieNode*)node, itemset));
        addTreeToCache(node_data->left, itemset_left, cache);
//        if(itemset.size() == 3 and itemset.at(0) == 16 and itemset.at(1) == 38 and itemset.at(2) == 40) {
//            cout << "data after left " << node_data << endl;
//        }

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

        Itemset itemset_right = addItem(itemset, item(node_data->test, POS_ITEM));
        pair<Node *, bool> res_right = cache->insert(itemset_right);
        Node *node_right = res_right.first;
        if (res_right.second) {
//            node_right->data = new TrieNodeData();
//            *((TrieNodeData*)node_right->data) = *(node_data->right);
            node_right->data = new TrieNodeData(*(((DepthTwo_NodeData*)node_data)->right));
        }
        else *((TrieNodeData*)node_right->data) = *(node_data->right);
//        node_data->right = node_right;
//        ((TrieNode*)node_right)->search_parents.push_back((TrieNode*)node);
//        ((TrieNode*)node_right)->search_parents.insert(make_pair((TrieNode*)node, itemset));
        addTreeToCache(node_data->right, itemset_right, cache);
//        if(itemset.size() == 3 and itemset.at(0) == 16 and itemset.at(1) == 38 and itemset.at(2) == 40) {
//            cout << "data after right " << node_data << endl;
//        }

//        if(itemset_right.size() == 5 and itemset_right.at(0) == 16 and itemset_right.at(1) == 28 and itemset_right.at(2) == 40 and itemset_right.at(3) == 43 and itemset_right.at(4) == 46) {
//            cout << "\nparchild-(";
//            printItemset(itemset, true, true);
//            cout << node_data->test << endl;
//            printItemset(itemset_right, true);
//            exit(0);
//        }

//        if (node_data->test >= 0) node_data->test *= -1;
//        cache->updateParents(node, node_left, node_right);
//        cache->updateParents(node, node_left, node_right, itemset);
//        node_data->test *= -1;
    }
}

void addTreeToCache(Node* node, NodeDataManager* ndm, Cache* cache, Depth depth) {
    
    auto* node_data = (CoverNodeData*)node->data;
    // cout << "toc " << node_data << endl;

    // cout << "coucouoo " << ((CoverNodeData*)node_data)->left <<  endl;

    if ( ((DepthTwo_NodeData*) (node_data->left)) != nullptr ) {
        //  cout << "left exists. split feat is " << node_data->test << endl;

        ndm->cover->intersect(node_data->test, NEG_ITEM);
        pair<Node *, bool> res_left = cache->insert(ndm, depth);
        Node *node_left = res_left.first;
        if (res_left.second) {
           node_left->data = new CoverNodeData();
           *((CoverNodeData*)node_left->data) = *((DepthTwo_NodeData*)(node_data->left));
            // node_left->data = new CoverNodeData(*((DepthTwo_NodeData*)(node_data->left)));
        }
        else *((CoverNodeData*)node_left->data) = *((DepthTwo_NodeData*)(node_data->left));
        // cout << "left insert is " << node_left << " it's data is " << node_left->data << " -- " << node_left->data->test << ":" << node_left->data->error << endl;
        node_data->left = (HashCoverNode*)node_left;
        // cout << "left set is " << node_data->left << " it's data is " << node_data->left->data << " -- " << node_data->left->data->test << ":" << node_data->left->data->error << endl;
        // cout << "bo " << ((CoverNodeData*)node_data)->left << endl;
        // cout << "bo " << ((CoverNodeData*)node_data)->left->data->test << endl;
        addTreeToCache(node_left, ndm, cache, depth + 1);
        ndm->cover->backtrack();

        // cout << "right exists. split feat is " << node_data->test << endl;
        ndm->cover->intersect(node_data->test, POS_ITEM);
        pair<Node *, bool> res_right = cache->insert(ndm, depth);
        Node *node_right = res_right.first;
        if (res_right.second) {
//            node_right->data = new CoverNodeData();
//            *((CoverNodeData*)node_right->data) = *(((DepthTwo_NodeData*)node_data)->right);
            node_right->data = new CoverNodeData(*((DepthTwo_NodeData*)(node_data->right)));
        }
        else *((CoverNodeData*)node_right->data) = *((DepthTwo_NodeData*)(node_data->right));
        //  cout << "right insert is " << node_right << " it's data is " << node_right->data << " -- " << node_right->data->test << ":" << node_right->data->error << endl;
        ((CoverNodeData*)node_data)->right = (HashCoverNode*)node_right;
        // cout << "right set is " << node_data->right << " it's data is " << node_data->right->data << " -- " << node_data->right->data->test << ":" << node_data->right->data->error << endl;
        addTreeToCache(node_right, ndm, cache, depth + 1);
        ndm->cover->backtrack();

        //cache->updateParents(node, node_left, node_right);
    }
    // else {
    //      cout << "no more child" << endl;
    // }
}


/*void addTreeToCache(Node* node, NodeDataManager* ndm, Cache* cache){
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
}*/

/*void setIteme(NodeData* node_data, const Itemset & itemset, Cache* cache){
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
}*/


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
    // if (ub <= lb and node->data->test >= 0){ // infeasible case. Avoid computing useless solution
        node->data = nodeDataManager->initData(); // no need to update the error
        Logger::showMessageAndReturn("infeasible case. ub = ", ub, " lb = ", lb);
        return node->data->error;
    }

    // The fact to not bound the search make it find the best solution in any case and remove the chance to recall this
    // function for the same node with a higher upper bound. Since this function is not exponential, we can afford that.
    ub = FLT_MAX;
//    Logger::showMessageAndReturn("nono ", node->data->leafError);

    // get the support and the support per class of the root node
    ErrorVals root_sup_clas = copyErrorVals(cover->getErrorValPerClass());
    Support root_sup = cover->getSupport();

    // update the next candidates list by removing the one already added
//    Logger::showMessage("list666: ");
//    printItemset(attributes_to_visit);
    vector<Attribute> attr;
    attr.reserve(attributes_to_visit.size() - 1);
//    if (cache != nullptr and node->data->test < 0) attr.push_back((node->data->test * -1) - 1);
    for(const auto attribute : attributes_to_visit) {
        if (last_added == attribute) continue;
//        if (attribute == last_added or (cache != nullptr and node->data->test < 0 and attribute == attr.at(0)) ) continue;
        attr.push_back(attribute);
    }
//    vector<Attribute> attr;
//
//    if (node->data->test < 0) {
//        int best_root = node->data->test * -1 - 1, best_left = -1, best_right = -1;
//        Node* best_left_node = cache->get(addItem(itemset, item(best_root, NEG_ITEM)));
//        Node* best_right_node = cache->get(addItem(itemset, item(best_root, POS_ITEM)));
//        if (best_left_node != nullptr and best_left_node->data != nullptr and best_left_node->data->test != INT32_MAX) best_left = best_left_node->data->test >= 0 ? best_left_node->data->test : best_left_node->data->test * -1 - 1;
//        if (best_right_node != nullptr and best_right_node->data != nullptr and best_right_node->data->test != INT32_MAX) best_right = best_right_node->data->test >= 0 ? best_right_node->data->test : best_right_node->data->test * -1 - 1;
//
//        Logger::showMessageAndReturn("anc test: ", (node->data->test * -1) - 1);
//
//        if (best_left != -1 and best_right != -1) {
//            attr.reserve(3);
//            attr.push_back(best_root);
//            attr.push_back(best_left);
//            attr.push_back(best_right);
//        }
//        else {
//            attr.reserve(attributes_to_visit.size() - 1);
//            attr.push_back(best_root);
//            if (best_left != -1) attr.push_back(best_left);
//            if (best_right != -1) attr.push_back(best_right);
//            int n_found = attr.size();
//            for(const auto attribute : attributes_to_visit) {
//                if ( attribute == last_added or attribute == attr.at(0) or (n_found == 2 and attribute == attr.at(1)) ) continue;
//                attr.push_back(attribute);
//            }
//        }
//    }
//    else {
//        attr.reserve(attributes_to_visit.size() - 1);
//        for(const auto attribute : attributes_to_visit) {
//        if (last_added == attribute) continue;
//            attr.push_back(attribute);
//        }
//    }



//    Logger::showMessage("list attrs: ");
//    printItemset(attr);
    Logger::showMessageAndReturn("lowerbound ", lb);

    // compute the different support per class we need to perform the search. Only a few mandatory are computed. The remaining are derived from them
    // matrix for supports per class
    auto **sups_sc = new ErrorVals* [attr.size()];
//    unique_ptr<unique_ptr<ErrorVal>> sups_sc(unique_ptr<ErrorVal>(new ErrorVals) [attr.size()]);
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
//    auto* best_tree = new TreeTwo();

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
//        ErrorVals igsc = newErrorVals(); //here
        unique_ptr<ErrorVal, ErrorvalsDeleter> igsc(newErrorVals()); //here
//        forEachClass ( i ) igsc.get()[i] = root_sup_clas[i] - idsc[i];
        subErrorVals(root_sup_clas, idsc, igsc.get());
        Support igs = root_sup - ids;

        // the feature tested as root is invalid since its two children cannot fulfill the minsup constraint
        if (igs < searcher->minsup or ids < searcher->minsup) {
            Logger::showMessageAndReturn("root impossible de splitter...on backtrack");
//            delete feat_best_tree;
//            deleteErrorVals(igsc);
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
//                delete feat_best_tree;
//                deleteErrorVals(igsc);
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
//                    deleteErrorVals(igjdsc);
                    continue; // test new left
                }

                Logger::showMessageAndReturn("le left testé peut splitter. on le regarde");
                LeafInfo ev2 = nodeDataManager->computeLeafInfo(igjdsc.get());
                Logger::showMessageAndReturn("le left a droite produit une erreur de ", ev2.error);

                // upper bound constraint is violated
                if (ev2.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur gauche du left montre rien de bon. best root: ", best_tree->root_data->error, " error found: ", ev2.error, " Un autre left...");
//                    deleteErrorVals(igjdsc);
                    continue; // test new left
                }

//                ErrorVals igjgsc = newErrorVals();
                unique_ptr<ErrorVal, ErrorvalsDeleter> igjgsc(newErrorVals());
                subErrorVals(igsc.get(), igjdsc.get(), igjgsc.get());
                LeafInfo ev1 = nodeDataManager->computeLeafInfo(igjgsc.get());
                Logger::showMessageAndReturn("le left a gauche produit une erreur de ", ev1.error);

                if (ev1.error + ev2.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur du left = ", ev1.error + ev2.error, " est pire que l'erreur du root existant (", feat_ub, "). Un autre left...");
//                    deleteErrorVals(igjdsc);
//                    deleteErrorVals(igjgsc);
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
//                    deleteErrorVals(idjgsc);
                    break;
                }

//                deleteErrorVals(igjdsc);

//                deleteErrorVals(igjgsc);
            }

            // there is no left child coupled to the root to produce lower error than the best tree so far. No need to look at right
            if (floatEqual(feat_best_tree->root_data->left->error, FLT_MAX)){
                Logger::showMessageAndReturn("aucun left n'a su améliorer l'arbre existant: ", feat_best_tree->root_data->left->error, "on garde l'ancien arbre");
//                deleteErrorVals(igsc);
//                delete feat_best_tree;
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
//                delete feat_best_tree;
//                deleteErrorVals(igsc);
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
//                best_tree->replaceTree(feat_best_tree);
                best_tree = move(feat_best_tree);
//                deleteErrorVals(igsc);
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
//                    deleteErrorVals(idjgsc);
                    continue; // test next right
                }

                Logger::showMessageAndReturn("le right testé peut splitter. on le regarde");
                LeafInfo ev1 = nodeDataManager->computeLeafInfo(idjgsc.get());
                Logger::showMessageAndReturn("le right a gauche produit une erreur de ", ev1.error);

                if (ev1.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur gauche du right montre rien de bon. Un autre right...");
//                    deleteErrorVals(idjgsc);
                    continue; // test next right
                }

                LeafInfo ev2 = nodeDataManager->computeLeafInfo(idjdsc);
                Logger::showMessageAndReturn("le right a droite produit une erreur de ", ev2.error);

                if (ev1.error + ev2.error >= feat_ub) {
                    Logger::showMessageAndReturn("l'erreur du right = ", ev1.error + ev2.error, " n'ameliore pas l'existant. Un autre right...");
//                    deleteErrorVals(idjgsc);
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
//                    deleteErrorVals(idjgsc);
                    break;
                }

//                deleteErrorVals(idjgsc);
            }

            // there is no left child coupled to the root and left to produce lower error than the best tree so far.
            if (floatEqual(feat_best_tree->root_data->right->error, FLT_MAX)){
//                cout << "pas de right convainquant" << endl;
                Logger::showMessageAndReturn("pas d'arbre mieux que le meilleur jusque là.");
//                deleteErrorVals(igsc);
//                delete feat_best_tree;
                continue; // test new root
            }

            feat_best_tree->root_data->error = feat_best_tree->root_data->left->error + feat_best_tree->root_data->right->error;
            feat_best_tree->root_data->size = feat_best_tree->root_data->left->size + feat_best_tree->root_data->right->size + 1;
//            best_tree->replaceTree(feat_best_tree);
            best_tree = move(feat_best_tree);
//            deleteErrorVals(igsc);
            Logger::showMessageAndReturn("ce triplet (root, left, right) ci donne une meilleure erreur que les précédents triplets: (", best_tree->root_data->test, ",", best_tree->root_data->left->left ? best_tree->root_data->left->test : -best_tree->root_data->left->test, ",", best_tree->root_data->right->left ? best_tree->root_data->right->test : -best_tree->root_data->right->test, ") err:", best_tree->root_data->error);
            if (floatEqual(best_tree->root_data->error, lb)) {
                Logger::showMessageAndReturn("The best tree is found");
                break;
            }
        }

        // deleteErrorVals(igsc);
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
//        delete best_tree;
        return best_error;
    }

    if (best_tree->root_data->test != INT32_MAX) {

        Logger::showMessageAndReturn("best tree found (root, left, right): (", best_tree->root_data->test, ",", best_tree->root_data->left->left ? best_tree->root_data->left->test : -best_tree->root_data->left->test, ",", best_tree->root_data->right->left ? best_tree->root_data->right->test : -best_tree->root_data->right->test, ") err:", best_tree->root_data->error);

        if (best_tree->root_data->size == 3 and best_tree->root_data->left->test == best_tree->root_data->right->test and floatEqual(best_tree->root_data->leafError, best_tree->root_data->left->error + best_tree->root_data->right->error)) {
//            cout << "pra" << endl;
            best_tree->root_data->size = 1;
            best_tree->root_data->error = best_tree->root_data->leafError;
            best_tree->root_data->test = best_tree->root_data->right->test;
//            delete best_tree->root_data->left;
//            best_tree->root_data->left = nullptr;
//            delete best_tree->root_data->right;
//            best_tree->root_data->right = nullptr;
            Logger::showMessageAndReturn("best twotree error = ", to_string(best_tree->root_data->error));
//            *(node->data) = *(best_tree->root_data);
            if (cachecover){
                *((CoverNodeData*)node->data) = *(best_tree->root_data);
                ((CoverNodeData*)node->data)->left = nullptr;
                ((CoverNodeData*)node->data)->right = nullptr;
            }
            else {
                *((TrieNodeData*)node->data) = *(best_tree->root_data);
            }

//            node->data = (NodeData *)best_tree->root_data;
//            best_tree->root_data = nullptr;
//            delete best_tree;
            return node->data->error;
        }

        if (cachecover) {
            // cout << "dsffi " << best_tree->root_data->left << endl;
            // cout << "node test " << ((CoverNodeData*)node->data)->test << " " << node << " " << (CoverNodeData*)node->data << endl;
            *((CoverNodeData*)node->data) = *(best_tree->root_data);
            // cout << "node test " << ((CoverNodeData*)node->data)->test << " " << node << " " << (CoverNodeData*)node->data << endl;


            // auto* node_data = node->data;

    // cout << "qzdefcoucouoo " << ((CoverNodeData*)node_data)->left <<  endl;

//            *(node->data) = *(best_tree->root_data);
//            node->data = (NodeData *) best_tree->root_data;
//            best_tree->root_data = nullptr;

            // if (itemset.size() == 4 and itemset.at(0) == 4 and itemset.at(1) == 29 and itemset.at(2) == 35 and itemset.at(3) == 47) {
            //     cout << "check what is happening" << endl;
            // }

            addTreeToCache(node, nodeDataManager, cache, searcher->maxdepth - 2 + 1);
            // cout << "rt " << node->data->test << endl;
            // cout << "node test " << ((CoverNodeData*)node->data)->test << " " << node << " " << (CoverNodeData*)node->data << endl;
            // cout << "rt " << ((CoverNodeData*)node->data)->left << endl;
            // cout << "rt " << ((CoverNodeData*)node->data)->left->data->test << endl;

            // if (itemset.size() == 4 and itemset.at(0) == 4 and itemset.at(1) == 29 and itemset.at(2) == 35 and itemset.at(3) == 47) {
            //     cout << "check what is happening after add" << endl;
            // }

        }
        else {
            if (cache->maxcachesize > NO_CACHE_LIMIT and cache->getCacheSize() + ((searcher->maxdepth + 1) * 4) > cache->maxcachesize) {
//                cout << "wipe_in" << endl;
                cache->wipe();
            }
//            cout << "azer" << endl;
//            cout << best_tree->root_data->test << " " << best_tree->root_data->left->data->test << " " << best_tree->root_data->right->data->test << endl;
            *((TrieNodeData*)node->data) = *(best_tree->root_data);
//            *(node->data) = *(best_tree->root_data);
//            node->data = (NodeData *) best_tree->root_data;
//            best_tree->root_data = nullptr;
//            cout << "\niteeeeemmmmeee ";
//            printItemset(itemset, true);
//            cout << "lowerrrr:" << lb << " error:" << best_tree->root_data->error << " root:" << best_tree->root_data->test << " left:" << best_tree->root_data->left->data->test << " l_error:" << best_tree->root_data->left->data->error << " right:" << best_tree->root_data->right->data->test << " r_error:" << best_tree->root_data->right->data->error << endl;
//cout << "azerty"  << endl;
addTreeToCache(best_tree->root_data, itemset, cache);
            Logger::showMessageAndReturn("tre: ", node->data->test);



        }
//        delete best_tree;

        Logger::showMessageAndReturn("best twotree error = ", to_string(node->data->error));

        return node->data->error;
    } else {
        // not root fulfill minsup constraint
//        delete best_tree;
//cout << "boss" << endl;
        node->data->error = node->data->leafError;
        return node->data->error;
    }

}