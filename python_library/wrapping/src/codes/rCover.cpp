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
    support = dm->getNTransactions();
}

RCover::RCover(bitset<M> *bitset1, int nword):dm(nullptr) {
    nWords = nword;
    coverWords = new stack<bitset<M>>[nWords];
    validWords = new int[nWords];
    for (int i = 0; i < nWords; ++i) {
        stack<bitset<M>> rword;
        rword.push(bitset1[i]);
        coverWords[i] = rword;
        validWords[i] = i;
    }
    int climit = nWords;
//    for (int j = 0; j < climit; ++j) {
//        if (coverWords[validWords[j]].top().none()){
//            int tmp = validWords[climit-1];
//            validWords[climit-1] = validWords[j];
//            validWords[j] = tmp;
//            --climit;
//            --j;
//        }
//    }
    limit.push(climit);
}

bitset<M>* RCover::getTopBitsetArray(){
    bitset<M>* tmp = new bitset<M>[nWords];
    for (int j = 0; j < nWords; ++j) {
        tmp[j] = coverWords[j].top();
    }
    return tmp;
}

void RCover::intersect1(Attribute attribute, bool positive) {
    //cout << "intersect" << endl;
    int climit = limit.top();
    //cout << "limit = " << climit << endl;
    support = 0;
    for (int i = 0; i < climit; ++i) {
        bitset<M> word;
        if (positive)
            word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else
            word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);
        support += word.count();

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

void RCover::intersect(Attribute attribute, bool positive) {
    //cout << "intersect" << endl;
    int climit = limit.top();
    //cout << "limit = " << climit << endl;
    sup_class = newSupports();
    zeroSupports(sup_class);
    support = 0;
    for (int i = 0; i < climit; ++i) {
        bitset<M> word;
        if (positive)
            word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else
            word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);

        int c = word.count();
        support += c;
        if (nclasses == 2){
            int addzero = (word & dm->getClassCover(0)[validWords[i]]).count();
            sup_class[0] += addzero;
            sup_class[1] += c - addzero;
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
    //cout << "done" << endl;
}


Supports RCover::intersectAndClass(Attribute attribute, bool positive) {
    Supports sc = newSupports(); zeroSupports(sc);
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);
        if (nclasses == 2){
            int addzero = (word & dm->getClassCover(0)[validWords[i]]).count();
            sc[0] += addzero; sc[1] += word.count() - addzero;
        } else forEachClass(n) sc[n] += (word & dm->getClassCover(n)[validWords[i]]).count();
    }
    return sc;
}


int RCover::intersectAndSup(Attribute attribute, bool positive) {
    int sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);
        sup += word.count();
    }
    return sup;
}


void RCover::intersectAndFillAll(Supports* row, vector<Attribute>& attributes, int start) {
    if (start >= attributes.size()) return;
    bitset<M> word;
    for (int i = 0; i < limit.top(); ++i) {
//        cout << "is = " << i << endl;
        for (int j = start; j < attributes.size(); ++j) {
            if (i == 0){
                row[j] = newSupports();
                zeroSupports(row[j]);
            }
            word = coverWords[validWords[i]].top() & dm->getAttributeCover(attributes[j])[validWords[i]];
            forEachClass(n) row[j][n] += (word & dm->getClassCover(n)[validWords[i]]).count();
//            cout << "test : " << row[j][1] << endl;
        }
    }
    //cout << "test : " << row[start][1] << endl;
}


void RCover::minus(bitset<M>* cover1) {
    //cout << "intersect" << endl;
    int climit = limit.top();
    //cout << "limit = " << climit << endl;
    for (int i = 0; i < climit; ++i) {
        coverWords[validWords[i]].push(coverWords[validWords[i]].top() & ~(cover1[validWords[i]]) );

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

Support RCover::minusMee(bitset<M>* cover1) {
    int tmpValid[nWords];
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
    Support toreturn = getSupport();
    backtrack();
    limit.pop();
    for (int i = 0; i < nWords; ++i) {
        coverWords[i].pop();
        validWords[i] = tmpValid[i];
    }
    return toreturn;
}

Supports RCover::minusMe(bitset<M>* cover1) {
    int tmpValid[nWords];
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
    Supports toreturn = getSupportPerClass();
    backtrack();
    limit.pop();
    for (int i = 0; i < nWords; ++i) {
        coverWords[i].pop();
        validWords[i] = tmpValid[i];
    }
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

Supports RCover::getSupportPerClass(){
    if (sup_class != nullptr) return sup_class;
    Supports itemsetSupport;
    itemsetSupport = newSupports();
    zeroSupports(itemsetSupport);
    if (nclasses == 2){
        bitset<M> * classCover = dm->getClassCover(0);
        int sum = 0;
        for (int i = 0; i < limit.top(); ++i) {
            sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
        }
        itemsetSupport[0] = sum;
        itemsetSupport[1] = getSupport() - sum;
    }
    else{
        for (int j = 0; j < nclasses; ++j) {
            bitset<M> * classCover = dm->getClassCover(j);
            int sum = 0;
            for (int i = 0; i < limit.top(); ++i) {
                sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
            }
            itemsetSupport[j] = sum;
        }
    }
    sup_class = itemsetSupport;
    return itemsetSupport;
}

/*pair<Supports, Support> RCover::getSupportPerClass(){
    bitset<M>** cover = new bitset<M>*[limit.top()];
    Supports supclass = newSupports();
    //zeroSupports(coverSupport.second);
    if (nclasses == 2){
        bitset<M> * classCover = dm->getClassCover(0);
        int sum = 0;
        for (int i = 0; i < limit.top(); ++i) {
            cover[i] = coverWords[validWords[i]].top() & classCover[validWords[i]];
            sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
        }
        itemsetSupport.first[0] = sum;
        itemsetSupport.second = getSupport();
        itemsetSupport.first[1] = itemsetSupport.second - sum;
    }
    else{
        for (int j = 0; j < nclasses; ++j) {
            bitset<M> * classCover = dm->getClassCover(j);
            int sum = 0;
            for (int i = 0; i < limit.top(); ++i) {
                sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
            }
            itemsetSupport.first[j] = sum;
            itemsetSupport.second += sum;
        }
    }

    return itemsetSupport;
}*/

Supports RCover::getClassSupport(){
    Supports classSupport = new int[dm->getNClasses()];
    for (int i = 0; i < dm->getNClasses(); ++i) {
        bitset<M>* cc = dm->getClassCover(i);
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