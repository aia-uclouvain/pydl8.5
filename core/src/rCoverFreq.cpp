//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCoverFreq.h"





RCoverFreq::RCoverFreq(DataManager *dmm): RCover(dmm) {}

void RCoverFreq::intersect(Attribute attribute, bool positive) {
    int climit = limit.top();
    sup_class = zeroSupports();
    support = 0;
    for (int i = 0; i < climit; ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);

        int word_sup = word.count();
        support += word_sup;
        if (nclasses == 2){
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

pair<Supports, Support> RCoverFreq::temporaryIntersect(Attribute attribute, bool positive) {
    Supports sc = zeroSupports();
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        int word_sup = word.count(); sup += word_sup;
        if (nclasses == 2){
            int addzero = (word & dm->getClassCover(0)[validWords[i]]).count();
            sc[0] += addzero; sc[1] += word_sup - addzero;
        } else forEachClass(n) {
            sc[n] += (word & dm->getClassCover(n)[validWords[i]]).count();
        }
    }
    return make_pair(sc, sup);
}



Supports RCoverFreq::getSupportPerClass(){
    if (sup_class != nullptr) return sup_class;
    sup_class = zeroSupports();
    if (nclasses == 2){
        bitset<M> * classCover = dm->getClassCover(0);
        int sum = 0;
        for (int i = 0; i < limit.top(); ++i) {
            sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
        }
        sup_class[0] = sum;
        sup_class[1] = getSupport() - sum;
    }
    else{
        for (int j = 0; j < nclasses; ++j) {
            bitset<M> * classCover = dm->getClassCover(j);
            for (int i = 0; i < limit.top(); ++i) {
                sup_class[j] += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
            }
        }
    }
    return sup_class;
}

Supports RCoverFreq::getSupportPerClass(bitset<M>** cover, int nValidWords, int* validIndexes){
    Supports sc = zeroSupports();
    for (int j = 0; j < nclasses; ++j) {
        bitset<M> * classCover = dm->getClassCover(j);
        for (int i = 0; i < nValidWords; ++i) {
            sc[j] += (*cover[i] & classCover[i]).count();
        }
    }
    return sc;
}

SupportClass RCoverFreq::countSupportClass(bitset<64> &coverWord, int wordIndex) {
    return coverWord.count();
}


/*RCoverFreq::RCoverFreq(DataManager *dmm): RCover(dmm) {}

void RCoverFreq::intersect(Attribute attribute, bool positive) {
    int current_limit = limit.top();
    sup_class = zeroSupports();
    support = 0;
    for (int i = 0; i < current_limit; ++i) {
        ulong word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);

        int word_sup = countSetBits(word);
        support += word_sup;
        if (nclasses == 2){
            int addzero = countSetBits(word & dm->getClassCover(0)[validWords[i]]);
            sup_class[0] += addzero;
            sup_class[1] += word_sup - addzero;
        } else forEachClass(n) sup_class[n] += countSetBits(word & dm->getClassCover(n)[validWords[i]]);

        if (word == 0UL){
            int tmp = validWords[current_limit-1];
            validWords[current_limit-1] = validWords[i];
            validWords[i] = tmp;
            --current_limit;
            --i;
        }
    }
    limit.push(current_limit);
}


pair<Supports, Support> RCoverFreq::temporaryIntersect(Attribute attribute, bool positive) {
    Supports sc = zeroSupports();
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        ulong word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        int word_sup = countSetBits(word); sup += word_sup;
        if (nclasses == 2){
            int addzero = countSetBits(word & dm->getClassCover(0)[validWords[i]]);
            sc[0] += addzero; sc[1] += word_sup - addzero;
        } else forEachClass(n) {
                sc[n] += countSetBits(word & dm->getClassCover(n)[validWords[i]]);
            }
    }
    return make_pair(sc, sup);
}


Supports RCoverFreq::getSupportPerClass(){
    if (sup_class != nullptr) return sup_class;
    sup_class = zeroSupports();
    if (nclasses == 2){
        ulong * classCover = dm->getClassCover(0);
        int sum = 0;
        for (int i = 0; i < limit.top(); ++i) {
            sum += countSetBits(coverWords[validWords[i]].top() & classCover[validWords[i]]);
        }
        sup_class[0] = sum;
        sup_class[1] = getSupport() - sum;
    }
    else{
        for (int j = 0; j < nclasses; ++j) {
            ulong * classCover = dm->getClassCover(j);
            for (int i = 0; i < limit.top(); ++i) {
                sup_class[j] += countSetBits(coverWords[validWords[i]].top() & classCover[validWords[i]]);
            }
        }
    }
    return sup_class;
}

Supports RCoverFreq::getSupportPerClass(ulong * cover, int nValidWords, int* validIndexes){
    Supports sc = zeroSupports();
    for (int j = 0; j < nclasses; ++j) {
        ulong * classCover = dm->getClassCover(j);
        for (int i = 0; i < nValidWords; ++i) {
            sc[j] += countSetBits(cover[i] & classCover[i]);
        }
    }
    return sc;
}

SupportClass RCoverFreq::countSupportClass(ulong coverWord, int wordIndex) {
    return countSetBits(coverWord);
}*/