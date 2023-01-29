//
// Created by Gael Aglin on 2019-12-23.
//

#include "dataManager.h"

DataManager::DataManager(ErrorVals supports, Transaction _ntransactions, Attribute _nattributes, Class _nclasses, Bool *data, Class *target): supports(supports), ntransactions_(_ntransactions), nattributes_(_nattributes), nclasses_(_nclasses) {

    nclasses_ = (nclasses_ == 1) ? 2 : nclasses_;
    nWords = (int)ceil((float)ntransactions_/M);
    b = new bitset<M> *[nattributes_];
    c = new bitset<M> *[nclasses_];

    for (int i = 0; i < nattributes_; i++){
        bitset<M> * attrCov = new bitset<M>[nWords];
        for (int j = 0; j < nWords; ++j) {
            int current_index = -1;
            Bool *start = data + (ntransactions_*i) + (M*j), *end;
            if (j != nWords - 1) end = data + (ntransactions_ * i) + (M*j) + M;
            else end = data + (ntransactions_ * i) + ntransactions_;

            int dist;
            auto itr = find(start, end, 1);
            while (itr != end && start < end) {
                dist = distance(start, itr);
                current_index += 1 + dist;
                attrCov[nWords-(j+1)].set(current_index);
                start += (dist + 1);
                itr = find(start, end, 1);
            }
        }
        b[i] = attrCov;
    }


    if (target != nullptr){
        for (int i = 0; i < nclasses_; i++){
            bitset<M> * classCov = new bitset<M>[nWords];
            for (int j = 0; j < nWords; ++j) {
                int current_index = -1;
                Class *start = target + (M*j), *end;
                if (j != nWords - 1) end = target + (M*j) + M;
                else end = target + ntransactions_;

                int dist;
                auto itr = find(start, end, i);
                while (itr != end && start < end) {
                    dist = distance(start, itr);
                    current_index += 1 + dist;
                    classCov[nWords-(j+1)].set(current_index);
                    start += (dist + 1);
                    itr = find(start, end, i);
                }
            }
            c[i] = classCov;
        }
    }
    else c = nullptr;


    GlobalParams::getInstance()->nattributes = nattributes_;
    GlobalParams::getInstance()->nclasses = nclasses_;
    GlobalParams::getInstance()->ntransactions = ntransactions_;
}

bitset<M>* DataManager::getAttributeCover(Attribute attr) {
    return b[attr];
}

bitset<M>* DataManager::getClassCover(Class clas) {
    return c[clas];
}