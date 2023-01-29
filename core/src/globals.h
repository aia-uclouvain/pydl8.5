#ifndef GLOBALS_H
#define GLOBALS_H

#include <climits>
#include <cfloat>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <chrono>

// type created for decision taken on an attribute (feature)
typedef int Bool;
//typedef unsigned char Bool;
// a class value
typedef int Class;
//typedef float Class;
//typedef unsigned char Class;
// a transaction id
typedef int Transaction;
// a feature number
typedef int Attribute;
//typedef unsigned short int Attribute;
// number of nodes in the tree
typedef int Size;
//typedef unsigned short int Size;
// depth of the tree
typedef int Depth;
//typedef unsigned short int Depth;
// error of the tree
typedef float Error;
// an item is a decision on an attribute (selected or not). n_items = 2 * n_attributes
typedef int Item;
//typedef unsigned short int Item;
// number of transactions covered by an itemset
typedef int Support;
// weighted support for a class
typedef float ErrorVal;
// array of supports per class
typedef ErrorVal* ErrorVals;
typedef const ErrorVal* constErrorVals;
//typedef Support *Supports;
typedef unsigned long ulong;
typedef std::vector<Item> Itemset;
typedef std::vector<Attribute> Attributes;

/*class GlobalParams {
public:
    static GlobalParams* getInstance() {
        static GlobalParams* instance = new GlobalParams();
        return instance;
    }
    Class nclasses;
    Attribute nattributes;
    Transaction ntransactions;
    bool verbose;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::string dataname;
    std::string out;

private:
    explicit GlobalParams() {
        std::cout << "params instanciated" << std::endl;
        verbose = false;
        startTime = std::chrono::high_resolution_clock::now();
        nclasses = -1;
        nattributes = -1;
        ntransactions = -1;
        dataname = "";
        out = "";
    }
};*/

class GlobalParams {
public:
    static GlobalParams* getInstance() {
        if(instance == nullptr) {
            instance = new GlobalParams();
        }
        return instance;
    }
    static void free() {
        delete instance;
        instance = nullptr;
    }
    Class nclasses;
    Attribute nattributes;
    Transaction ntransactions;
    bool verbose;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::string dataname;
    std::string out;
    static GlobalParams* instance;


    GlobalParams(const GlobalParams&) = delete;

private:
    explicit GlobalParams() {
        verbose = false;
        startTime = std::chrono::high_resolution_clock::now();
        nclasses = -1;
        nattributes = -1;
        ntransactions = -1;
        dataname = "";
        out = "";
    }
};




/*class GlobalParams {
public:
    static GlobalParams* getInstance() {
        if (instance == nullptr)
            instance = new GlobalParams();
        return instance;
    }
    Class nclasses;
    Attribute nattributes;
    bool verbose;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::string dataname;
    std::string out;
    static GlobalParams* instance;
    void set_nclasses(Class nclasses) { this->nclasses = nclasses; }
private:
    explicit GlobalParams() = default;

};*/




//extern Class nclasses;
//extern Attribute nattributes;
//extern bool verbose;
//extern std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
//extern std::string dataname;
//extern std::string out;
//
//std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
//Class nclasses;
//Attribute nattributes;
//bool verbose = false;
//std::string dataname;
//std::string out;


#define NO_SUP INT_MAX // SHRT_MAX
#define NO_ERR FLT_MAX
#define NO_CACHE_LIMIT 0
#define NEG_ITEM 0
#define POS_ITEM 1
#define NO_GAIN FLT_MAX
#define NO_ITEM INT_MAX // SHRT_MAX
#define NO_ATTRIBUTE INT_MAX // SHRT_MAX
#define NO_DEPTH INT_MAX
#define ZERO 0.f


// compute item value based on the attribute and its decision value
#define item(attribute, value) ( attribute * 2 + value )
// compute the attribute value based on the item value
#define item_attribute(item) ( item / 2 )
// compute the decision on an attribute based on its item value
#define item_value(item) ( item % 2 )
// loop in each class value
#define forEachClass(n) for ( Class n = 0; n < GlobalParams::getInstance()->nclasses; ++n )
// loop in each index in an array
#define forEach(index, array) for ( int index = 0; index < array.size; ++index )
// redefine a class name to make it short
#define TFND TrieFreq_NodeData*
// redefine a class name to make it short
#define FNDM Freq_NodeDataManager*


// create (dynamic allocation of vector of size = number of classes)
ErrorVals newErrorVals();

// create (dynamic allocation of vector of size = number of classes) and fill vector of support with zeros
ErrorVals zeroErrorVals();

// fill vector of supports passed in parameter with zeros
void zeroErrorVals(ErrorVals supports);

// free the memory
void deleteErrorVals(ErrorVals supports);
//void deleteErrorVals(constErrorVals supports);

// copy values of support array src to dest
void copyErrorVals(constErrorVals src, ErrorVals dest);

// create support array dest, copy values of array in parameter in dest and return dest
ErrorVals copyErrorVals(ErrorVals supports);

// return sum of value of support
ErrorVal sumErrorVals(constErrorVals supports);

// return dest which is array of addition of src2 from src1
void addErrorVals(constErrorVals src1, constErrorVals src2, ErrorVals dest);

// return dest which is array of substraction of src2 from src1
void subErrorVals(constErrorVals src1, constErrorVals src2, ErrorVals dest);

ErrorVals subErrorVals(ErrorVals src1, ErrorVals src2);

bool floatEqual(float f1, float f2);

void merge(const Itemset &src1, const Itemset &src2, Itemset &dest);

void addItem(const Itemset &src, Item item, Itemset &dest);

Itemset addItem(const Itemset &src, Item item, bool quiet = true);

void printItemset(const Itemset &itemset, bool force = false, bool newline = true);

std::string custom_to_str(float val);


#endif