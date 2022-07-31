//
// Created by Gael Aglin on 2019-12-23.
//

#ifndef RCOVER_H
#define RCOVER_H

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
    stack<bitset<M>, vector<bitset<M>>>* coverWords;
    int* validWords;
    stack<int, vector<int>> limit;
    int nWords;
    DataManager* dm;
    ErrorVals sup_class = nullptr;
    int support = -1;

    RCover(DataManager* dmm);

    RCover(RCover&& cover) noexcept ;

    virtual ~RCover(){
        delete[] coverWords;
        delete[] validWords;
        delete [] sup_class;
    }

    virtual void intersect(Attribute attribute, bool positive = true) = 0;

    virtual pair<ErrorVals, Support> temporaryIntersect(Attribute attribute, bool positive = true) = 0;

    Support temporaryIntersectSup(Attribute attribute, bool positive = true);

    ErrorVal getDiffErrorVal(bitset<M>* cover1, int* valids, int nvalids, bool cover_is_first = false);

    ErrorVals getDiffErrorVals(bitset<M>* cover1, bool cover_is_first = false);

    bitset<M>* getDiffCover(bitset<M>* cover1, bool cover_is_first = false);

    bitset<M>* getTopCover() const;

    Support getSupport();

    virtual ErrorVals getErrorValPerClass() = 0;

    virtual ErrorVal getErrorVal(bitset<M>& coverWord, int wordIndex) = 0;

    virtual ErrorVals getErrorValPerClass(bitset<M>* cover, int nValidWords, int* validIndexes) = 0;

    void backtrack();

    void print();

    string outprint();

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
                if (index == -1)
                    wordIndex = container->dm->getNClasses();
                else if (index == 0){
                    wordIndex = index;
                }
            }
        }

        explicit iterator() : wordIndex(-1), container(nullptr) {}

        unsigned int getFirstSetBitPos(unsigned long n) {
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
                    return container->sup_class[wordIndex];
                }
            }

        }

        self_type operator++() {
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

    };

    iterator begin(bool trans_loop = false) {
        return iterator(this, 0, trans_loop);
    }

    iterator end(bool trans_loop = false) {
        return iterator(this, -1, trans_loop);
    }

};

// custom specialization of std::hash can be injected in namespace std
namespace std {
    template<>
    struct hash<RCover> {
        std::size_t operator()(const RCover& array) const noexcept {
            std::size_t h = array.nWords;
            for (int i = 0; i < array.nWords; ++i) {
                h ^= array.coverWords[i].top().to_ulong() + 0x9e3779b9 + 64 * h + h / 4;
            }
            return h;
        }
    };

    template<>
    struct equal_to<RCover> {
        bool operator()(const RCover& lhs, const RCover& rhs) const noexcept {
            for (int i = 0; i < lhs.nWords; ++i) {
                if (lhs.coverWords[i].top().to_ulong() != rhs.coverWords[i].top().to_ulong()) return false;
            }
            return true;
        }
    };
}


#endif //RCOVER_H
