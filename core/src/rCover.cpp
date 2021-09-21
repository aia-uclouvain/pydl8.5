//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCover.h"

bool isZero(float x){
    return x == 0.0;
}

RCover::RCover(DataManager *dmm, vector<float>* weights):dm(dmm) {
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
            /*if (weights){
                int size = dm->getNTransactions()%M;
                vector<float> tmp = vector<float>(weights->begin(), weights->begin() + size);
                auto it = tmp.begin();
                while ((it = std::find_if(it, tmp.end(), isZero)) != tmp.end())
                {
                    int ind = distance(tmp.begin(), it);
                    word.set(size - ind - 1, false);
                    it++;
                }
            }*/
        }
        else {
            /*if (weights){
                vector<float> tmp = vector<float>(weights->begin() + i * M, weights->begin() + ((i+1) * M - 1));
                auto it = tmp.begin();
                while ((it = std::find_if(it, tmp.end(), isZero)) != tmp.end())
                {
                    int ind = distance(tmp.begin(), it);
                    word.set(M - ind - 1, false);
                    it++;
                }
            }*/
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
Supports RCover::minusMe(bitset<M>* cover1) {
    int maxValidNumber = limit.top();
    bitset<M>** difcover = new bitset<M>*[maxValidNumber];
    int* validIndexes = new int[maxValidNumber];
    int nvalid = 0;
    for (int i = 0; i < maxValidNumber; ++i) {
        bitset<M> potential_word = cover1[validWords[i]] & ~coverWords[validWords[i]].top();
        if (!potential_word.none()){
            difcover[nvalid] = &potential_word;
            validIndexes[nvalid] = validWords[i];
            ++nvalid;
        }
    }

    Supports toreturn = getSupportPerClass(difcover, nvalid, validIndexes);
    delete [] difcover;
    delete [] validIndexes;
    return toreturn;
}

SupportClass RCover::countDif(bitset<M>* cover1) {
    SupportClass sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> potential_word = cover1[validWords[i]] & ~coverWords[validWords[i]].top();
        if (!potential_word.none()){
            sup += countSupportClass(potential_word, validWords[i]);
        }
    }
    return sup;
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
    deleteSupports(sup_class);
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