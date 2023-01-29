//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCoverFreq.h"

RCoverFreq::RCoverFreq(DataManager *dmm): RCover(dmm) {}

void RCoverFreq::intersect(Attribute attribute, bool positive) {
    int climit = limit.top();
    deleteErrorVals(sup_class);
    sup_class = zeroErrorVals();
    support = 0;
    for (int i = 0; i < climit; ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);

        int word_sup = word.count();
        support += word_sup;
        if (GlobalParams::getInstance()->nclasses == 2){
            int addzero = (word & dm->getClassCover(0)[validWords[i]]).count();
            sup_class[0] += addzero;
            sup_class[1] += word_sup - addzero;
        } else forEachClass(n) sup_class[n] += (word & dm->getClassCover(n)[validWords[i]]).count();

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

pair<ErrorVals, Support> RCoverFreq::temporaryIntersect(Attribute attribute, bool positive) {
    ErrorVals sc = zeroErrorVals();
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        int word_sup = word.count(); sup += word_sup;
        if (GlobalParams::getInstance()->nclasses == 2){
            int addzero = (word & dm->getClassCover(0)[validWords[i]]).count();
            sc[0] += addzero; sc[1] += word_sup - addzero;
        } else forEachClass(n) {
            sc[n] += (word & dm->getClassCover(n)[validWords[i]]).count();
        }
    }
    return make_pair(sc, sup);
}


ErrorVals RCoverFreq::getErrorValPerClass(){
    if (sup_class != nullptr) return sup_class;
    sup_class = zeroErrorVals();
    if (GlobalParams::getInstance()->nclasses == 2){
        bitset<M> * classCover = dm->getClassCover(0);
        int sum = 0;
        for (int i = 0; i < limit.top(); ++i) {
            sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
        }
        sup_class[0] = sum;
        sup_class[1] = getSupport() - sum;
    }
    else{
        for (int j = 0; j < GlobalParams::getInstance()->nclasses; ++j) {
            bitset<M> * classCover = dm->getClassCover(j);
            for (int i = 0; i < limit.top(); ++i) {
                sup_class[j] += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
            }
        }
    }
    return sup_class;
}

ErrorVals RCoverFreq::getErrorValPerClass(bitset<M>* cover, int nValidWords, int* validIndexes){
    ErrorVals sc = zeroErrorVals();
    for (int j = 0; j < GlobalParams::getInstance()->nclasses; ++j) {
        bitset<M> * classCover = dm->getClassCover(j);
        for (int i = 0; i < nWords; ++i) {
            sc[j] += (cover[i] & classCover[i]).count();
        }
    }
    //forEachClass(i) cout << sc[i] << ",";
    return sc;
}

// count the support for the word
ErrorVal RCoverFreq::getErrorVal(bitset<64> &coverWord, int wordIndex) {
    return coverWord.count();
}