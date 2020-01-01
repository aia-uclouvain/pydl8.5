//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RSBS_RCOVER_H
#define RSBS_RCOVER_H

#include <stack>
#include <vector>
#include <bitset>
#include <iostream>
#include <utility>
#include "globals.h"
#include "dataManager.h"
#include <cmath>

using namespace std;

#define M 64

class RCover {

public:
    stack<bitset<M>>* coverWords;
    int* validWords;
    stack<int> limit;
    int nWords;
    DataManager* dm;
    int* sup = nullptr;

    RCover(DataManager* dmm);

    ~RCover(){
        delete[] coverWords;
        delete[] validWords;
    }

    void intersect(Attribute attribute, bool positive = true);

    int getSupport();

    pair<Supports, Support> getSupportPerClass();

    int* getClassSupport();

    vector<int> getTransactionsID();

    void backtrack();

    void print();

    class iterator {
    public:
        typedef iterator self_type;
        typedef int value_type;
        typedef int &reference;
        typedef int *pointer;
        typedef std::input_iterator_tag iterator_category;
        typedef int difference_type;

        explicit iterator(RCover *container_, size_t index = 0, bool trans = false) : container(container_) {
            if (trans){
                trans_loop = true;
                if (index == -1)
                    wordIndex = container->limit.top();
                else if (index == 0){
                    wordIndex = index;
                    pos = 0;
                    transInd = 0;
                    first = true;
                    word = container->coverWords[container->validWords[0]].top();
                    setNextTransID();
                }

            } else{
                trans_loop = false;
                if (index == 0){
                    wordIndex = index;
                    //sup = container->getSupportPerClass().first;
                }
                else if (index == -1)
                    wordIndex = container->dm->getNClasses();
            }
        }

        explicit iterator() : wordIndex(-1), container(nullptr) {}

        int getFirstSetBitPos(long n)
        {
            return log2(n & -n) + 1;
        }

        void setNextTransID() {
            if (wordIndex < container->limit.top()) {
                int indexForTransactions = container->nWords - (container->validWords[wordIndex]+1);
                int pos = getFirstSetBitPos(word.to_ulong());

                if (pos >= 1){
                    if (first){
                        transInd = pos - 1;
                        first = false;
                    }
                    else
                        transInd += pos;

                    value = indexForTransactions * M + transInd;
                    word = (word >> pos);
                } else{
                    ++wordIndex;
                    transInd = 0;
                    first = true;
                    if (wordIndex < container->limit.top()){
                        word = container->coverWords[container->validWords[wordIndex]].top();
                        setNextTransID();
                    }
                }
            }

        }

        value_type operator*() const {
            if (trans_loop){
                if (wordIndex >= container->limit.top()){
                    throw std::out_of_range("Out of Range Exception!");
                }
                else {
                    return value;
                }
            } else{
                if (wordIndex >= container->dm->getNClasses()){
                    throw std::out_of_range("Out of Range Exception!");
                }
                else {
                    return container->sup[wordIndex];
                }
            }

        }

        self_type operator++() {
            //cout << "bbbbarrive" << endl;
            if (trans_loop)
                setNextTransID();
            else
                ++wordIndex;
            return *this;
        }

        bool operator==(const self_type rhs) {
            return container + trans_loop + wordIndex == rhs.container + rhs.trans_loop + rhs.wordIndex;
        }

        bool operator!=(const self_type rhs) {
            return container + trans_loop + wordIndex != rhs.container + rhs.trans_loop + rhs.wordIndex;
        }

        RCover *container;
        int wordIndex;
        int pos;
        int value;
        int transInd;
        bool first;
        bitset<M> word;
        bool trans_loop;
        //int ntrans = -1;
        //int alltransInd;

    };

    iterator begin(bool trans_loop = false)
    {
        return iterator(this, 0, trans_loop);
    }

    iterator end(bool trans_loop = false)
    {
        return iterator(this, -1, trans_loop);
    }

};


#endif //RSBS_RCOVER_H
