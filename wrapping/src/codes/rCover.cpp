//
// Created by Gael Aglin on 2019-12-23.
//

#include "rCover.h"
#include <cmath>

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
}

void RCover::intersect(Attribute attribute, bool positive) {
    //cout << "intersect" << endl;
    int climit = limit.top();
    //cout << "limit = " << climit << endl;
    for (int i = 0; i < climit; ++i) {
        if (positive)
            coverWords[validWords[i]].push(coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]]);
        else
            coverWords[validWords[i]].push(coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]) );

        if (coverWords[validWords[i]].top().none()){
            int tmp = validWords[climit-1];
            validWords[climit-1] = validWords[i];
            validWords[i] = tmp;
            --climit;
            --i;
        }
    }
    limit.push(climit);
    //cout << "done" << endl;
}

int RCover::getSupport() {
    int sum = 0;
    for (int i = 0; i < limit.top(); ++i) {
        sum += coverWords[validWords[i]].top().count();
    }
    return sum;
}

pair<Supports, Support> RCover::getSupportPerClass(){
    pair<Supports, Support> itemsetSupport;
    itemsetSupport.first = newSupports();
    zeroSupports(itemsetSupport.first);
    for (int j = 0; j < nclasses; ++j) {
        bitset<M> * classCover = dm->getClassCover(j);
        int sum = 0;
        for (int i = 0; i < limit.top(); ++i) {
            sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
        }
        itemsetSupport.first[j] = sum;
        itemsetSupport.second += sum;
    }
    return itemsetSupport;
}

int* RCover::getClassSupport(){
    int* classSupport = new int[dm->getNClasses()];
    for (int i = 0; i < dm->getNClasses(); ++i) {
        bitset<M> * cc = dm->getClassCover(i);
        int sup = 0;
        for (int j = 0; j < nWords; ++j) {
            sup += cc[j].count();
        }
        classSupport[i] = sup;
    }
    return classSupport;
}

/*vector<int> RCover::getTransactionsID() {
    vector<int> tid;
    for (int i = 0; i < limit.top(); ++i) {
        int indexForTransactions = nWords - (validWords[i]+1);
        bitset<M> word = coverWords[validWords[i]].top();
        int pos = getFirstSetBitPos(word.to_ulong());
        int transInd = pos - 1;
        while (pos >= 0){
            tid.push_back(indexForTransactions * M + transInd );
            word = (word >> pos);
            pos = getFirstSetBitPos(word.to_ulong());
            transInd += pos;
        }
    }
    return tid;
}*/

void RCover::backtrack() {
    limit.pop();
    int climit = limit.top();
    for (int i = 0; i < climit; ++i) {
        coverWords[validWords[i]].pop();
    }
}

void RCover::print() {
    for (int i = 0; i < nWords; ++i) {
        cout << coverWords[i].top() << " ";
    }
    cout << endl;
}