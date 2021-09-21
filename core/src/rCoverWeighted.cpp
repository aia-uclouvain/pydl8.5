//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCoverWeighted.h"


RCoverWeighted::RCoverWeighted(DataManager *dmm, vector<float>* weights):RCover(dmm, weights), weights(weights) {}

RCoverWeighted::RCoverWeighted(RCoverWeighted &&cover, vector<float>* weights): RCover(move(cover)), weights(weights) {}

void RCoverWeighted::intersect(Attribute attribute, bool positive) {
    int climit = limit.top();
    sup_class = zeroSupports();
    support = 0;
    for (int i = 0; i < climit; ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);

        int real_word_index = nWords - (validWords[i]+1);
        forEachClass(n) {
            bitset<M> intersectedWord = word & dm->getClassCover(n)[validWords[i]];
            /*vector<int>&& tids = getTransactionsID(intersectedWord, real_word_index);
            for (auto tid : tids) {
                support++;
                sup_class[n] += (*weights)[tid];
            }*/
            pair<SupportClass, Support>&& r = getSups(intersectedWord, real_word_index);
            support += r.second;
            sup_class[n] += r.first;
        }

        if (word.none()){
            int tmp = validWords[climit-1];
            validWords[climit-1] = validWords[i];
            validWords[i] = tmp;
            --climit;
            --i;
        }
    }
    limit.push(climit);
}

/**
 * temporaryIntersect - compute a temporary intersection of the current cover with an item
 * to get its support and support per class. No changes are performed in the current cover
 * this function is only used to prepare data for computation of the specific algo for 2-depth trees
 * @param attribute - the attribute to interszect with
 * @param positive - the item of the attribute
 * @return a pair of support per class and support
 */
pair<Supports, Support> RCoverWeighted::temporaryIntersect(Attribute attribute, bool positive) {
    Supports sc = zeroSupports();
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        int real_word_index = nWords - (validWords[i]+1);
        forEachClass(n) {
            bitset<M> intersectedWord = word & dm->getClassCover(n)[validWords[i]];
            /*vector<int>&& tids = getTransactionsID(intersectedWord, real_word_index);
            for (auto tid : tids) {
                sup++;
                sc[n] += (*weights)[tid];
            }*/
            pair<SupportClass, Support>&& r = getSups(intersectedWord, real_word_index);
            sup += r.second;
            sc[n] += r.first;
//            if ( sc[n] < 0) cout << "sup neg" << endl;
        }
    }
//    cout << sc[0] << " " << sc[1] << endl;
    return make_pair(sc, sup);
}


Supports RCoverWeighted::getSupportPerClass(){
    if (sup_class) {
//        cout << sup_class[0] << " " << sup_class[1] << endl;
        return sup_class;
    }
    sup_class = zeroSupports();
    for (int j = 0; j < nclasses; ++j) {
        bitset<M> * classCover = dm->getClassCover(j);
        for (int i = 0; i < limit.top(); ++i) {
            // get the real index of the word
            int real_word_index = nWords - (validWords[i] + 1);
            bitset<M> intersectedWord = coverWords[validWords[i]].top() & classCover[validWords[i]];
            /*vector<int>&& tids = getTransactionsID(intersectedWord, real_word_index);
            for (auto tid : tids) {
                sup_class[j] += (*weights)[tid];
            }*/
            pair<SupportClass, Support>&& r = getSups(intersectedWord, real_word_index);
            sup_class[j] += r.first;
//            if ( sup_class[j] < 0) cout << "sup neg" << endl;
        }
    }
//    cout << sup_class[0] << " " << sup_class[1] << endl;
    return sup_class;
}

Supports RCoverWeighted::getSupportPerClass(bitset<M>** cover, int nValidWords, int* validIndexes){
    Supports sc = zeroSupports();
    for (int j = 0; j < nclasses; ++j) {
        bitset<M> * classCover = dm->getClassCover(j);
        for (int i = 0; i < nValidWords; ++i) {
            // get the real index of the word
            int real_word_index = nWords - (validIndexes[i] + 1);
            bitset<M> intersectedWord = *cover[i] & classCover[i];
            pair<SupportClass, Support>&& r = getSups(intersectedWord, real_word_index);
            sc[j] += r.first;
        }
    }
    return sc;
}

SupportClass RCoverWeighted::countSupportClass(bitset<64> &coverWord, int wordIndex) {
    int real_word_index = nWords - (wordIndex + 1);
    return getSups(coverWord, real_word_index).first;
}

/**
 * isKthBitSet - function to check wheter the bit at a specific index
 * is 1 or not. The first index is 1
 * @param number - decimal value of the bitset
 * @param index - the index value to check
 * @return boolean value representing the result of the query
 */
bool isKthBitSet(u_long number, int index) {
    return (number & (1 << (index - 1))) != 0;
}

/**
 * getFirstSetBitPos - get the index of the first bit set in a binary number
 * remember that index goes from right to left and the first index is 1
 * @param number - int value of the binary number
 * @return the index of the first set bit
 */
unsigned int getFirstSetBitPos(const u_long& number) { return log2(number & -number) + 1;}

/**
 * getTransactionsID
 * @return the list of transactions in the current cover
 */
vector<int> RCoverWeighted::getTransactionsID() {
    vector<int> tid;
    for (int i = 0; i < limit.top(); ++i) {
        int indexForTransactions = nWords - (validWords[i]+1);
        bitset<M> word = coverWords[validWords[i]].top();
        u_long w = word.to_ulong();
        int pos = getFirstSetBitPos(w);
        int transInd = pos - 1;
        while (pos >= 1){
            tid.push_back(indexForTransactions * M + transInd);
            word = (word >> pos);
            w = word.to_ulong();
            pos = getFirstSetBitPos(w);
            transInd += pos;
        }
    }
    return tid;
}

/**
 * getTransactionsID - get list of transactions given an specific word and its index in the words array
 * @param word
 * @param real_word_index
 * @return the list of transactions
 */
vector<int> RCoverWeighted::getTransactionsID(bitset<M>& word, int real_word_index) {

    vector<int> tid;
    int pos = getFirstSetBitPos(word.to_ulong());
    int transInd = pos - 1;

    while (pos >= 1){
        tid.push_back(real_word_index * M + transInd );
        word = (word >> pos);
        pos = getFirstSetBitPos(word.to_ulong());
        transInd += pos;
    }
    return tid;
}


pair<SupportClass, Support> RCoverWeighted::getSups(bitset<M>& word, int real_word_index){
    pair<SupportClass, Support> result(0, 0);
    int pos = getFirstSetBitPos(word.to_ulong());
    int transInd = pos - 1;

    while (pos >= 1){
//        if (real_word_index * M + transInd >= 2597) cout << "mémoire dépassée" << endl;
//        if ((*weights)[real_word_index * M + transInd] < 0) cout << "ind = " << (real_word_index * M + transInd) << " val = " << (*weights)[real_word_index * M + transInd] << endl;
        result.first += (*weights)[real_word_index * M + transInd];
//        cout << (*weights)[real_word_index * M + transInd] << ", ";
        //sup[0] = 4.62428e-44, sup[1] = 0.730769 sum = 0.730769 maxclassval = 0.730769 error = 0 class = 1
        result.second++;
        word = (word >> pos);
        pos = getFirstSetBitPos(word.to_ulong());
        transInd += pos;
    }
//    if (result.first != 0) cout << endl;
    return result;
}
