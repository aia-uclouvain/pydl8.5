#ifndef GLOBALS_H
#define GLOBALS_H

#include <climits>
#include <cfloat>

// type created for decision taken on an attribute (feature)
typedef int Bool;
// a class value
typedef int Class;
// a transaction id
typedef int Transaction;
// a feature number
typedef int Attribute;
// number of nodes in the tree
typedef int Size;
// depth of the tree
typedef int Depth;
// error of the tree
typedef float Error;
// an item is a decision on an attribute (selected or not). n_items = 2 * n_attributes
typedef int Item;
// number of transactions covered by an itemset
typedef int Support;
// array of supports per class
typedef int* Supports;

#define NO_SUP INT_MAX // SHRT_MAX
#define NO_ERR FLT_MAX
#define NO_GAIN FLT_MAX
#define NO_ITEM INT_MAX // SHRT_MAX
#define NO_DEPTH INT_MAX

// compute item value based on the attribute and its decision value
#define item(attribute, value) ( attribute * 2 + value )
// compute the attribute value based on the item value
#define item_attribute(item) ( item / 2 )
// compute the decision on an attribute based on its item value
#define item_value(item) ( item % 2 )

#include <iostream>
#include <map>
#include <iterator>

// the array is a light-weight vector that does not do copying or resizing of storage space.
template<class A>
struct Array {
public:
    A* elts; //the "array" of elements
    int size; //the real size of elements in the array while "allocsize" is the allocated size

    Array () { }

    Array(A* elts, int size) {
        this->elts = elts;
        this->size = size;
    }

    Array(int allocsize, int size) {
        this->size = size;
        elts = new A[allocsize];
    }

    //~Array() {} //destructor does not do anything. Make sure you call free method after using the object

    void alloc(int size_) {
        this->size = size_;
        elts = new A[size_];
    }

    void free() {
        delete[] elts;
    }

    void resize(int size_) { // we don't know its original size any more
        this->size = size_;
    }

    void push_back(A elt) { // we can walk out of allocated space
        elts[size] = elt;
        ++size;
    }

    int getSize(){
        return size;
    }

    A &operator[](int i) { return elts[i]; }
};


void merge(Array<Item> src1, Array<Item> src2, Array<Item> dest);

void addItem(Array<Item> src1, Item item, Array<Item> dest);

#define forEach(index, array) for ( int index = 0; index < array.size; ++index )

// create (dynamic allocation of vector of size = number of classes)
Supports newSupports();

// create (dynamic allocation of vector of size = number of classes) and fill vector of support with zeros
Supports zeroSupports();

// fill vector of supports passed in parameter with zeros
void zeroSupports(Supports supports);

// free the memory
void deleteSupports(Supports supports);

// copy values of support array src to dest
void copySupports(Supports src, Supports dest);

// create support array dest, copy values of array in parameter in dest and return dest
Supports copySupports(Supports supports);

// return sum of value of support
Support sumSupports(Supports supports);

// return dest which is array of substraction of src2 from src1
void minSupports(Supports src1, Supports src2, Supports dest);

// return dest which is array of addition of src2 from src1
void plusSupports(Supports src1, Supports src2, Supports dest);

extern Class nclasses;
extern Attribute nattributes;
extern std::map<int, int> attrFeat;
extern bool verbose;

#define forEachClass(n) for ( Class n = 0; n < nclasses; ++n )

#endif