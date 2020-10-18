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
    limit.push(climit);
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

/*RCover::RCover() {}*/

bitset<M>* RCover::getTopBitsetArray() const{
    bitset<M>* tmp = new bitset<M>[nWords];
    for (int j = 0; j < nWords; ++j) {
        tmp[j] = coverWords[j].top();
    }
    return tmp;
}

/*void RCover::intersect(Attribute attribute, bool positive) {
    int climit = limit.top();
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
}*/


/*void RCover::weightedIntersect(Attribute attribute, const vector<float>& weights, bool positive) {
    int climit = limit.top();
    sup_class = zeroSupports();
    support = 0;
    for (int i = 0; i < climit; ++i) {
        bitset<M> word;
        if (positive)
            word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else
            word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        coverWords[validWords[i]].push(word);
        support += word.count();

        int real_word_index = nWords - (validWords[i]+1);
        forEachClass(n) {
                bitset<M> intersectedWord = word & dm->getClassCover(n)[validWords[i]];
                vector<int> tids = getTransactionsID(intersectedWord, real_word_index);
                for (auto tid : tids) {
                    sup_class[n] += weights[tid];
                }
            sup_class[n] += (word & dm->getClassCover(n)[validWords[i]]).count();
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
}*/

/**
 * temporaryIntersect - compute a temporary intersection of the current cover with an item
 * to get its support and support per class. No changes are performed in the current cover
 * this function is only used to prepare data for computation of the specific algo for 2-depth trees
 * @param attribute - the attribute to interszect with
 * @param positive - the item of the attribute
 * @return a pair of support per class and support
 */
/*pair<Supports, Support> RCover::temporaryIntersect(Attribute attribute, bool positive) {
    Supports sc = zeroSupports();
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        int c = word.count();
        sup += c;
        if (nclasses == 2){
            int addzero = (word & dm->getClassCover(0)[validWords[i]]).count();
            sc[0] += addzero; sc[1] += c - addzero;
        } else forEachClass(n) {
            sc[n] += (word & dm->getClassCover(n)[validWords[i]]).count();
        }
    }
    return make_pair(sc, sup);
}*/

/*pair<Supports, Support> RCover::temporaryWeightedIntersect(Attribute attribute, const vector<float>& weights, bool positive) {
    Supports sc = zeroSupports();
    Support sup = 0;
    for (int i = 0; i < limit.top(); ++i) {
        bitset<M> word;
        if (positive) word = coverWords[validWords[i]].top() & dm->getAttributeCover(attribute)[validWords[i]];
        else word = coverWords[validWords[i]].top() & ~(dm->getAttributeCover(attribute)[validWords[i]]);

        int real_word_index = nWords - (validWords[i]+1);
        forEachClass(n) {
            bitset<M> intersectedWord = word & dm->getClassCover(n)[validWords[i]];
            vector<int> tids = getTransactionsID(intersectedWord, real_word_index);
            for (auto tid : tids) {
                sup_class[n] += weights[tid];
            }
            sup_class[n] += (word & dm->getClassCover(n)[validWords[i]]).count();
        }
    }
    return make_pair(sc, sup);
}*/

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

RCover::RCover() {}

int RCover::getSupport() {
    if (support > -1) return support;
    int sum = 0;
    for (int i = 0; i < limit.top(); ++i) {
        sum += coverWords[validWords[i]].top().count();
    }
    support = sum;
    return sum;
}

/*Supports RCover::getSupportPerClass(){
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
            int sum = 0;
            for (int i = 0; i < limit.top(); ++i) {
                sum += (coverWords[validWords[i]].top() & classCover[validWords[i]]).count();
            }
            sup_class[j] = sum;
        }
    }
    return sup_class;
}*/

/*bool isKthBitSet(u_long n, int k) {
    if (n & (1 << (k - 1))) return true;
    else return false;
}*/

/**
 * getFirstSetBitPos - get the index of the first bit set in a binary number
 * remember that index goes from right to left and the first index is 0
 * @param number - int value of the binary number
 * @return the index of the first set bit
 */
/*unsigned int getFirstSetBitPos(int number) { return log2(number & -number) + 1;}*/

/**
 * getTransactionsID
 * @return the list of transactions in the current cover
 */
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

/**
 * getTransactionsID - get list of transactions given an specific word and its index in the words array
 * @param word
 * @param real_word_index
 * @return the list of transactions
 */
/*vector<int> RCover::getTransactionsID(bitset<M>& word, int real_word_index) {
    vector<int> tid;
    int pos = getFirstSetBitPos(word.to_ulong());
    int transInd = pos - 1;
    while (pos >= 0){
        tid.push_back(real_word_index * M + transInd );
        word = (word >> pos);
        pos = getFirstSetBitPos(word.to_ulong());
        transInd += pos;
    }
    return tid;
}*/

/*Supports RCover::getWeightedSupportPerClass(const vector<float>& weights) {
    if (sup_class) return sup_class;
    sup_class = zeroSupports();
    for (int j = 0; j < nclasses; ++j) {
        bitset<M> * classCover = dm->getClassCover(j);
        for (int i = 0; i < limit.top(); ++i) {
            // get the real index of the word
            int real_word_index = nWords - (validWords[i]+1);
            bitset<M> intersectedWord = coverWords[validWords[i]].top() & classCover[validWords[i]];
            vector<int> tids = getTransactionsID(intersectedWord, real_word_index);
            for (auto tid : tids) {
                sup_class[j] += weights[tid];
            }
        }
    }
    return sup_class;
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

/*void RCover::intersect1(Attribute attribute, bool positive) {
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
}*/


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


/*void RCover::intersectAndFillAll(Supports* row, vector<Attribute>& attributes, int start) {
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
}*/


/*void RCover::minus(bitset<M>* cover1) {
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
}*/


/*Support RCover::minusMee(bitset<M>* cover1) {
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
    Support toreturn = getSupport();
    backtrack();
    limit.pop();
    for (int i = 0; i < nWords; ++i) {
        coverWords[i].pop();
        validWords[i] = tmpValid[i];
    }
    delete [] tmpValid;
    return toreturn;
}*/


/*Supports RCover::getClassSupport(){
    return dm->getSupports();
}*/