//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCover.h"

RCover::RCover(DataManager *dmm):dm(dmm) {
    nWords = (int)ceil((float)dm->getNTransactions()/M);
    coverWords = new stack<bitset<M>>[nWords];
    cout << "cover basic answers after creating coverwords" << endl;
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

bitset<M>* RCover::getTopBitsetArray() const{
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
 * minusMe - this function computes the support per class for the cover c = cover1 - currentcover
 * @param cover1 - a cover to perform the minus operation
 * @return the support per class of the resultant cover
 */
Supports RCover::minusMe(bitset<M>* cover1, const vector<float>* weights) {
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
    Supports toreturn = getSupportPerClass(weights);
    backtrack();
    limit.pop();
    for (int i = 0; i < nWords; ++i) {
        coverWords[i].pop();
        validWords[i] = tmpValid[i];
    }
    delete [] tmpValid;
    return toreturn;
}

int RCover::getSupport() {
    if (support > -1) return support;
    int sum = 0;
    for (int i = 0; i < limit.top(); ++i) {
        sum += coverWords[validWords[i]].top().count();
    }
    support = sum;
    return sum;
}

void RCover::backtrack() {
    limit.pop();
    int climit = limit.top();
    for (int i = 0; i < climit; ++i) {
        coverWords[validWords[i]].pop();
    }
    support = -1;
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