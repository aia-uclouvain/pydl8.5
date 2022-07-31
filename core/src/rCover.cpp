//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCover.h"

RCover::RCover(DataManager *dmm):dm(dmm) {
    nWords = (int)ceil((float)dm->getNTransactions()/M);
    coverWords = new stack<bitset<M>, vector<bitset<M>>>[nWords];
    validWords = new int[nWords];
    for (int i = 0; i < nWords; ++i) {
        stack<bitset<M>, vector<bitset<M>>> rword;
        bitset<M> word;
        word.set();
        if(i == 0 && dm->getNTransactions()%M != 0){
            for (int j = dm->getNTransactions()%M; j < M; ++j) {
                word.set(j, false);
            }
        }
        rword.push(word);
        coverWords[i] = rword;
        validWords[i] = i;
    }
    limit.push(nWords);
    support = dm->getNTransactions();
}

RCover::RCover(RCover &&cover) noexcept {
    coverWords = cover.coverWords;
    validWords = cover.validWords;
    limit = cover.limit;
    nWords = cover.nWords;
    dm = cover.dm;
    sup_class = cover.sup_class;
    support= cover.support;
}

bitset<M>* RCover::getTopCover() const{
    auto* tmp = new bitset<M>[nWords];
    for (int j = 0; j < nWords; ++j) {
        tmp[j] = coverWords[j].top();
    }
    return tmp;
}

/**
 * temporaryIntersectSup - this function intersect the cover with an item just to
 * compute the support of the intersection. Nothing is change in the current cover
 * At the end of the function, the cover is still the same
 * @param attribute - the attribute to intersect with
 * @param positive -  the item of the attribute
 * @return the value of the support of the intersection
 */
Support RCover::temporaryIntersectSup(Attribute attribute, bool positive) {
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);
        sup += word.count();
    }
    return sup;
}

/**
 * getDiffErrorVals - this function computes the support per class for the cover c = cover1 - currentcover
 * @param cover1 - a cover to perform the minus operation
 * @return the support per class of the resultant cover
 */
ErrorVal RCover::getDiffErrorVal(bitset<M>* cover1, int* valids, int nvalids, bool cover_is_first) {
    ErrorVal err_val = 0;
    bitset<M> tmp_word;
    int my_limit = (cover_is_first) ? limit.top() : nvalids;
    int* my_valids = (cover_is_first) ? validWords : valids;
    for (int i = 0; i < my_limit; ++i) {
        if (cover_is_first) tmp_word = coverWords[my_valids[i]].top() & ~cover1[my_valids[i]];
        else tmp_word = cover1[my_valids[i]] & ~coverWords[my_valids[i]].top();
        if (tmp_word.any()) err_val += getErrorVal(tmp_word, my_valids[i]);
    }
    return err_val;
}

ErrorVals RCover::getDiffErrorVals(bitset<M>* cover1, bool cover_is_first) {
    int nvalid = 0;
    auto diff_cover = new bitset<M>[nWords];
    auto validIndexes = new int[nWords];
    for (int i = 0; i < nWords; ++i) {
        bitset<M> tmp_word;
        if (cover_is_first) tmp_word = coverWords[validWords[i]].top() & ~cover1[validWords[i]];
        else tmp_word = cover1[i] & ~coverWords[i].top();
        diff_cover[i] = tmp_word;
        validIndexes[nvalid] = i;
        ++nvalid;
    }
    ErrorVals err_vals = getErrorValPerClass(diff_cover, nvalid, validIndexes);
    delete [] diff_cover;
    delete [] validIndexes;
    return err_vals;
}

bitset<M>* RCover::getDiffCover(bitset<M>* cover1, bool cover_is_first) {
    int maxValidNumber = limit.top();
    auto diff_cover = new bitset<M>[maxValidNumber];
    for (int i = 0; i < nWords; ++i)
        if (cover_is_first) diff_cover[i] = coverWords[i].top() & ~cover1[i];
        else diff_cover[i] = cover1[i] & ~coverWords[i].top();
    return diff_cover;
}

int RCover::getSupport() {
    if (support > -1) return support;
    support = 0;
    for (int i = 0; i < limit.top(); ++i) support += coverWords[validWords[i]].top().count();
    return support;
}

void RCover::backtrack() {
    limit.pop();
    int climit = limit.top();
    for (int i = 0; i < climit; ++i) {
        coverWords[validWords[i]].pop();
    }
    support = -1;
    deleteErrorVals(sup_class);
    sup_class = nullptr;
}

void RCover::print() {
    for (int i = 0; i < nWords; ++i) {
        cout << coverWords[i].top().to_string() << " ";
    }
    cout << endl;
}

string RCover::outprint() {
    string s = "";
    for (int i = 0; i < nWords; ++i) {
        s += coverWords[i].top().to_string() + " ";
    }
    return s;
}