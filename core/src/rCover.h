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
    stack<bitset<M>, vector<bitset<M>>> *coverWords;
    int *validWords;
    stack<int, vector<int>> limit;
    int nWords;
    DataManager *dm;
    ErrorVals sup_class = nullptr;
    int support = -1;

    RCover(DataManager *dmm);

    RCover(RCover &&cover) noexcept;

    virtual ~RCover() {
        delete[] coverWords;
        delete[] validWords;
        delete[] sup_class;
    }

    virtual void intersect(Attribute attribute, bool positive = true) = 0;

    virtual pair<ErrorVals, Support> temporaryIntersect(Attribute attribute, bool positive = true) = 0;

    Support temporaryIntersectSup(Attribute attribute, bool positive = true);

    ErrorVal getDiffErrorVal(bitset<M> *cover1, int *valids, int nvalids, bool cover_is_first = false);

    ErrorVals getDiffErrorVals(bitset<M> *cover1, bool cover_is_first = false);

    bitset<M> *getDiffCover(bitset<M> *cover1, bool cover_is_first = false);

    bitset<M> *getTopCover() const;

    Support getSupport();

    virtual ErrorVals getErrorValPerClass() = 0;

    virtual ErrorVal getErrorVal(bitset<M> &coverWord, int wordIndex) = 0;

    virtual ErrorVals getErrorValPerClass(bitset<M> *cover, int nValidWords, int *validIndexes) = 0;

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

        explicit iterator(RCover *container_, size_t index = 0, bool tids = false) : container(container_) {
            if (tids) { // iterating over transactions in cover (coverWords)
                trans_iteration = true;
                if (index == -1) { // end() iterator
                    wordOrder = container->limit.top(); // wordOrder set to index after the last valid word
                }
                else if (index == 0) { // begin() iterator
                    wordOrder = index; // we start from the first valid word
                    currentWordView = container->coverWords[container->validWords[wordOrder]].top(); // get the first word
                    setNextTransID(); // prepare the iterator to return the first transaction
                }

            }
            else { // iterating over sup_class
                trans_iteration = false;
                if (index == -1) { // end() iterator
                    wordOrder = container->dm->getNClasses(); // wordOrder set to index after the last valid word
                }
                else if (index == 0) { // begin() iterator
                    wordOrder = index; // we start from the first valid word
                }
            }
        }

        explicit iterator() : wordOrder(-1), container(nullptr) {}

        static unsigned int getFirstSetBitPos(unsigned long long n) {
            return log2(n & -n) + 1;
        }

        void setNextTransID() {
            if (wordOrder >= container->limit.top()) return; // we are at the end of the coverWords
            int realWordIndex = container->nWords - (container->validWords[wordOrder] + 1);
            int transIDInWord = -1;
            int pos = getFirstSetBitPos(currentWordView.to_ullong());

            if (pos >= 1) { // there is a set bit in the word
                transIDInWord += pos;
                value = realWordIndex * M + transIDInWord;
                bitset<M> mask; // create a mask to get the bits after the first set bit
                mask.set();
                mask = (mask << pos);
                currentWordView = (currentWordView & mask);
            } else { // there is no set bit in the word. We move to the next word
                ++wordOrder;
                if (wordOrder < container->limit.top()) {
                    currentWordView = container->coverWords[container->validWords[wordOrder]].top(); // get the next word
                    setNextTransID();
                }
            }
        }

        value_type operator*() const {
            if (trans_iteration) {
                if (wordOrder >= container->limit.top()) {
                    throw std::out_of_range("Out of Range Exception!");
                } else {
                    return value;
                }
            } else {
                if (wordOrder >= container->dm->getNClasses()) {
                    throw std::out_of_range("Out of Range Exception!");
                } else {
                    return container->sup_class[wordOrder];
                }
            }
        }

        self_type operator++() {
            if (trans_iteration)
                setNextTransID();
            else
                ++wordOrder;
            return *this;
        }

        bool operator==(const self_type rhs) const {
            return container + trans_iteration + wordOrder == rhs.container + rhs.trans_iteration + rhs.wordOrder;
        }

        bool operator!=(const self_type rhs) const {
            return container + trans_iteration + wordOrder != rhs.container + rhs.trans_iteration + rhs.wordOrder;
        }

        RCover *container;
        int wordOrder; // order of the word being iterated
        int value; // value returned by the iterator (transaction id or a class support)
        bitset<M> currentWordView; // current word being iterated
        bool trans_iteration;

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
        std::size_t operator()(const RCover &array) const noexcept {
            std::size_t h = array.nWords;
            for (int i = 0; i < array.nWords; ++i) {
                h ^= array.coverWords[i].top().to_ullong() + 0x9e3779b9 + 64 * h + h / 4;
            }
            return h;
        }
    };

    template<>
    struct equal_to<RCover> {
        bool operator()(const RCover &lhs, const RCover &rhs) const noexcept {
            for (int i = 0; i < lhs.nWords; ++i) {
                if (lhs.coverWords[i].top().to_ullong() != rhs.coverWords[i].top().to_ullong()) return false;
            }
            return true;
        }
    };
}


#endif //RCOVER_H
