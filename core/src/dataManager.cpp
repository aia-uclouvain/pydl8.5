//
// Created by Gael Aglin on 2019-12-23.
//

#include "dataManager.h"

DataManager::DataManager(Supports supports, Transaction ntransactions, Attribute nattributes, Class nclasses, Bool *data, Class *target):supports(supports), ntransactions(ntransactions), nattributes(nattributes), nclasses(nclasses) {
    nclasses = (nclasses == 1) ? 2 : nclasses;
    nWords = (int)ceil((float)ntransactions/M);
    b = new bitset<M> *[nattributes];
    c = new bitset<M> *[nclasses];

    for (int i = 0; i < nattributes; i++){
        bitset<M> * attrCov = new bitset<M>[nWords];
        for (int j = 0; j < nWords; ++j) {
            int current_index = -1;
            Bool *start = data + (ntransactions*i) + (M*j), *end;
            if (j != nWords - 1) end = data + (ntransactions * i) + (M*j) + M;
            else end = data + (ntransactions * i) + ntransactions;

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


    if (target){
        for (int i = 0; i < nclasses; i++){
            bitset<M> * classCov = new bitset<M>[nWords];
            for (int j = 0; j < nWords; ++j) {
                int current_index = -1;
                Class *start = target + (M*j), *end;
                if (j != nWords - 1) end = target + (M*j) + M;
                else end = target + ntransactions;

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


    ::nattributes = nattributes;
    ::nclasses = nclasses;
}

bitset<M>* DataManager::getAttributeCover(Attribute attr) {
    return b[attr];
}

bitset<M>* DataManager::getClassCover(Class clas) {
    return c[clas];
}


/*DataManager::DataManager(Supports supports, Transaction ntransactions, Attribute nattributes, Class nclasses, Bool *data, Class *target):supports(supports), ntransactions(ntransactions), nattributes(nattributes), nclasses(nclasses) {
    nclasses = (nclasses == 1) ? 2 : nclasses;
    nWords = (int)ceil((float)ntransactions/M);
    b = new ulong *[nattributes];
    c = new ulong *[nclasses];

    for (int i = 0; i < nattributes; i++){
        ulong * attrCov = new ulong[nWords];
        for (int j = 0; j < nWords; ++j) {
            int currentindex = -1;
            Bool* start = data + (ntransactions*i) + (M*j);
            Bool* end = nullptr;
            if (j != nWords - 1) end = data + (ntransactions*i) + (M*j) + M;
            else end = data + (ntransactions*i) + ntransactions;

            int dist = 0;
            auto itr = find(start, end, 1);
            while (itr != end && start < end) {
                dist = distance(start, itr);
                currentindex += 1 + dist;
                set_1(attrCov[nWords-(j+1)], currentindex);
                start += (dist + 1);
                itr = find(start, end, 1);
            }
        }
        b[i] = attrCov;
        //cout << "attr : " << i << " word = " << attrCov->to_string() << endl;
    }


    if (target){
        for (int i = 0; i < nclasses; i++){
            ulong * classCov = new ulong [nWords];
            for (int j = 0; j < nWords; ++j) {
                int currentindex = -1;
                Class *start = target + (M*j), *end;
                if (j != nWords - 1) end = target + (M*j) + M;
                else end = target + ntransactions;

                int dist;
                auto itr = find(start, end, i);
                while (itr != end && start < end) {
                    dist = distance(start, itr);
                    currentindex += 1 + dist;
                    set_1(classCov[nWords-(j+1)], currentindex);
                    start += (dist + 1);
                    itr = find(start, end, i);
                }
            }
            c[i] = classCov;
        }
    }
    else c = nullptr;


    ::nattributes = nattributes;
    ::nclasses = nclasses;
}

ulong * DataManager::getAttributeCover(Attribute attr) {
    return b[attr];
}

ulong * DataManager::getClassCover(Class classe) {
    return c[classe];
}*/