#ifndef GLOBALS_H
#define GLOBALS_H

#include <climits>
#include <cfloat>

typedef int Bool;
typedef int Class;
typedef int Transaction;
typedef int Attribute;
typedef int Size; // size of a tree
typedef int Depth;
typedef float Error;
typedef Attribute Item; // an item is an attribute and its binary value
typedef Transaction Support;
typedef Support *Supports;

#define NO_SUP INT_MAX // SHRT_MAX
#define NO_ERR FLT_MAX
#define NO_GAIN FLT_MAX
#define NO_ITEM INT_MAX // SHRT_MAX
#define NO_DEPTH INT_MAX

#define item(attribute, value) ( attribute * 2 + value )
#define item_attribute(item) ( item / 2 )
#define item_value(item) ( item % 2 )

#include <iostream>
#include <map>
#include <iterator>

// the array is a light-weight vector that does not do copying or resizing of storage space.
template<class A>
struct Array {
public:

    Array () { }

    Array(A* e, int s) {
        elts = e;
        size = s;
    }

    Array(int allocsize, int size) {
        this->size = size;
        elts = new A[allocsize];
    }

    //~Array() {} //desctructor does not do anything. Make sure you call free method after using the object

    A *elts;
    int size;

    void alloc(int size) {
        this->size = size;
        elts = new A[size];
    }

    void free() {
        delete[] elts;
    }

    void resize(int size) { // we don't know its original size any more
        this->size = size;
    }

    void push_back(A a) { // we can walk out of allocated space
        elts[size] = a;
        ++size;
    }

    int getSize(){
        return size;
    }

    A &operator[](int i) { return elts[i]; }
};


void merge(Array<Item> src1, Array<Item> src2, Array<Item> dest);

void addItem(Array<Item> src1, Item item, Array<Item> dest);

#define forEach(i, a) for ( int i = 0; i < a.size; ++i )

// create (dynamic allocation of vector of size = number of classes)
Supports newSupports();

// create (dynamic allocation of vector of size = number of classes) and fill vector of support with zeros
Supports zeroSupports();

// fill vector of support with zeros
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
extern bool nps;
extern bool verbose;

#define forEachClass(n) for ( Class n = 0; n < nclasses; ++n )

#endif