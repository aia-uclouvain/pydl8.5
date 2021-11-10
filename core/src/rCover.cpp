//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCover.h"

RCover::RCover(DataManager *dmm):dm(dmm) {
    nWords = (int)ceil((float)dm->getNTransactions()/M);
    coverWords = new stack<bitset<M>>[nWords];
    validWords = new int[nWords];
    for (int i = 0; i < nWords; ++i) {
        stack<bitset<M>> rword;
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

/*Supports RCover::minusMe(bitset<M>* cover1) {
    int* tmpValid = new int[nWords];
    for (int j = 0; j < nWords; ++j) {
        tmpValid[j] = validWords[j];
    }
    int nvalid = 0;
    for (int i = 0; i < nWords; ++i) {
        coverWords[validWords[i]].push(~coverWords[validWords[i]].top() );

        if (!coverWords[validWords[i]].top().none()){
            validWords[nvalid] = i;
            ++nvalid;
        }
    }
    limit.push(nvalid);
    int climit = nvalid;
    for (int i = 0; i < climit; ++i) {
        coverWords[validWords[i]].push(coverWords[validWords[i]].top() & (cover1[validWords[i]]) );

        if (coverWords[validWords[i]].top().none()){
            int tmp = validWords[climit-1];
            validWords[climit-1] = validWords[i];
            validWords[i] = tmp;
            --climit;
            --i;
        }
    }
    limit.push(climit);
    deleteSupports(sup_class);
//    Supports tmp = sup_class;
    sup_class = nullptr;
    Supports toreturn = copySupports(getSupportPerClass());
    backtrack();
    limit.pop();
    for (int i = 0; i < nWords; ++i) {
        coverWords[i].pop();
        validWords[i] = tmpValid[i];
    }
    delete [] tmpValid;
//    sup_class = tmp;
    return toreturn;
}*/

ErrorVal RCover::getDiffErrorVal(bitset<M>* cover1, int* valids, int nvalids, bool cover_is_first) {
    ErrorVal err_val = 0;
    bitset<M> tmp_word;
//    for (int i = 0; i < limit.top(); ++i) {
//        if (cover_is_first) tmp_word = coverWords[validWords[i]].top() & ~cover1[validWords[i]];
//        else tmp_word = cover1[validWords[i]] & ~coverWords[validWords[i]].top();
//        if (tmp_word.any()) err_val += getErrorVal(tmp_word, validWords[i]);
//    }
//    for (int i = 0; i < nWords; ++i) {
//        if (cover_is_first) tmp_word = coverWords[i].top() & ~cover1[i];
//        else tmp_word = cover1[i] & ~coverWords[i].top();
//        if (tmp_word.any()) err_val += getErrorVal(tmp_word, i);
//    }
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
    int maxValidNumber = limit.top(), nvalid = 0;
    auto diff_cover = new bitset<M>[nWords];
//    auto validIndexes = new int[maxValidNumber];
    auto validIndexes = new int[nWords];
    for (int i = 0; i < nWords; ++i) {
        bitset<M> tmp_word;
        if (cover_is_first) tmp_word = coverWords[validWords[i]].top() & ~cover1[validWords[i]];
        else tmp_word = cover1[i] & ~coverWords[i].top();
//        if (tmp_word.any()){
//            diff_cover[nvalid] = tmp_word;
//            validIndexes[nvalid] = validWords[i];
//            ++nvalid;
//        }
        diff_cover[i] = tmp_word;
        validIndexes[nvalid] = i;
        ++nvalid;
    }
    ErrorVals err_vals = getErrorValPerClass(diff_cover, nvalid, validIndexes);
    delete [] diff_cover;
    delete [] validIndexes;
    return err_vals;
}

/*ErrorVals RCover::getDiffErrorVals(bitset<M>* cover1, bool cover_is_first) {
    int maxValidNumber = limit.top(), nvalid = 0;
    auto diff_cover = new bitset<M>[maxValidNumber];
    auto validIndexes = new int[maxValidNumber];
    for (int i = 0; i < maxValidNumber; ++i) {
        bitset<M> tmp_word;
        if (cover_is_first) tmp_word = coverWords[validWords[i]].top() & ~cover1[validWords[i]];
        else tmp_word = cover1[validWords[i]] & ~coverWords[validWords[i]].top();
        if (tmp_word.any()){
            diff_cover[nvalid] = tmp_word;
            validIndexes[nvalid] = validWords[i];
            ++nvalid;
        }
    }
    ErrorVals err_vals = getErrorValPerClass(diff_cover, nvalid, validIndexes);
    delete [] diff_cover;
    delete [] validIndexes;
    return err_vals;
}*/

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


/*
RCover::RCover(DataManager *dmm): dm(dmm) {
    nWords = (int)ceil((float)dm->getNTransactions()/M);
    coverWords = new stack<ulong>[nWords];
    validWords = new int[nWords];
    for (int i = 0; i < nWords; ++i) {
        stack<ulong> rword;
        ulong word = ULONG_MAX;
        if(i == 0 && dm->getNTransactions()%M != 0)
            for (int j = dm->getNTransactions()%M; j < M; ++j) set_0(word, j);
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

ulong* RCover::getTopBitsetArray() const{
    auto* tmp = new ulong [nWords];
    for (int j = 0; j < nWords; ++j) {
        tmp[j] = coverWords[j].top();
    }
    return tmp;
}

Support RCover::temporaryIntersectSup(Attribute attribute, bool positive) {
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        ulong word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);
        sup += countSetBits(word);
    }
    return sup;
}

Supports RCover::minusMe(ulong* cover) {
    int maxValidNumber = limit.top();
    ulong* diff_cover = new ulong[maxValidNumber];
    int* validIndexes = new int[maxValidNumber];
    int nvalid = 0;
    for (int i = 0; i < maxValidNumber; ++i) {
        ulong potential_word = cover[validWords[i]] & ~coverWords[validWords[i]].top();
        if (potential_word != 0UL){
            diff_cover[nvalid] = potential_word;
            validIndexes[nvalid] = validWords[i];
            ++nvalid;
        }
    }
    Supports to_return = getSupportPerClass(diff_cover, nvalid, validIndexes);
    delete [] diff_cover;
    delete [] validIndexes;
    return to_return;
}

SupportClass RCover::countDif(ulong* cover) {
    SupportClass sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        ulong potential_word = cover[validWords[i]] & ~coverWords[validWords[i]].top();
        if (potential_word != 0UL) sup += countSupportClass(potential_word, validWords[i]);
    }
    return sup;
}

int RCover::getSupport() {
    if (support > -1) return support;
    support = 0;
    for (int i = 0; i < limit.top(); ++i) support += countSetBits(coverWords[validWords[i]].top());
    return support;
}

void RCover::backtrack() {
    limit.pop();
    for (int i = 0; i < limit.top(); ++i) coverWords[validWords[i]].pop();
    support = -1;
    deleteSupports(sup_class);
    sup_class = nullptr;
}

string printWord(ulong word){
    string ret = "";
    ulong i = 1UL << (sizeof(word) * CHAR_BIT - 1);
    while (i > 0) {
        ret += (word & i) ? "1" : "0";
        i >>= 1;
    }
    return ret;
}

void RCover::print() {
    for (int i = 0; i < nWords; ++i) cout << printWord(coverWords[i].top()) << " ";
    cout << endl;
}

string RCover::outprint() {
    string s = "";
    for (int i = 0; i < nWords; ++i) s += printWord(coverWords[i].top()) + " ";
    return s;
}*/
