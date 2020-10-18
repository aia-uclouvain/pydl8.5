//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCoverWeighted.h"


RCoverWeighted::RCoverWeighted(DataManager *dmm):RCover(dmm) {}

RCoverWeighted::RCoverWeighted(RCoverWeighted &&cover) : RCover(move(cover)) {}

RCoverWeighted::RCoverWeighted(bitset<M> *bitset1, int nword):RCover(bitset1, nword) {}

void RCoverWeighted::intersect(Attribute attribute, const vector<float>* weights, bool positive) {
    int climit = limit.top();
    sup_class = zeroSupports();
    support = 0;
    for (int i = 0; i < climit; ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);
//        support += word.count();

        int real_word_index = nWords - (validWords[i]+1);
        forEachClass(n) {
            bitset<M> intersectedWord = word & dm->getClassCover(n)[validWords[i]];
            vector<int>&& tids = getTransactionsID(intersectedWord, real_word_index);
            for (auto tid : tids) {
                support++;
                sup_class[n] += (*weights)[tid];
            }
//            sup_class[n] += intersectedWord.count();
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
pair<Supports, Support> RCoverWeighted::temporaryIntersect(Attribute attribute, const vector<float>* weights, bool positive) {
    Supports sc = zeroSupports();
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

//        sup += word.count();
        int real_word_index = nWords - (validWords[i]+1);
        forEachClass(n) {
            bitset<M> intersectedWord = word & dm->getClassCover(n)[validWords[i]];
            vector<int>&& tids = getTransactionsID(intersectedWord, real_word_index);
            for (auto tid : tids) {
                sup++;
                sc[n] += (*weights)[tid];
            }
//            sc[n] += intersectedWord.count();
        }
    }
    return make_pair(sc, sup);
}


Supports RCoverWeighted::getSupportPerClass(const vector<float>* weights){
    if (sup_class) return sup_class;
    sup_class = zeroSupports();
    for (int j = 0; j < nclasses; ++j) {
        bitset<M> * classCover = dm->getClassCover(j);
        for (int i = 0; i < limit.top(); ++i) {
            // get the real index of the word
            int real_word_index = nWords - (validWords[i] + 1);
            bitset<M> intersectedWord = coverWords[validWords[i]].top() & classCover[validWords[i]];
            vector<int>&& tids = getTransactionsID(intersectedWord, real_word_index);
//            float ff = 0;
            for (auto tid : tids) {
                sup_class[j] += (*weights)[tid];
            }
//            sup_class[j] += intersectedWord.count();
            /*if (ff != intersectedWord.count()){
                cout << "ff = " << ff << ", sup = " << intersectedWord.count() << ", realind = " << real_word_index << ", word = " << intersectedWord << ", wordint = " << intersectedWord.to_ulong() << endl;
            }*/
        }
    }
    return sup_class;
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
//    cout << "word_trans : " << word << ", pos = " << pos << ", transid = " << transInd << endl;
//    int i = 0;
    while (pos >= 1){
        tid.push_back(real_word_index * M + transInd );
//        cout << "pos = " << (real_word_index * M + transInd) << " " << pos << endl;
        word = (word >> pos);
        pos = getFirstSetBitPos(word.to_ulong());
//        cout << word << " " << word.to_ulong() << ", p = " << pos << endl;
        transInd += pos;
//        cout << "ww : " << word << ", pos = " << pos << ", transid = " << transInd << endl;
//        i++;
//        if (i == 5) break;
    }
    return tid;
}
