#ifndef GLOBALS_H
#define GLOBALS_H

#include <climits>
#include <cfloat>
#include <iostream>
#include <map>
#include <iterator>
#include <thread>
#include <functional>
#include <algorithm>
#include <vector>
#include <cmath>

// type created for decision taken on an attribute (feature)
typedef int Bool;
//typedef unsigned char Bool;
// a class value
typedef int Class;
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
typedef float SupportClass;
// array of supports per class
typedef SupportClass *Supports;
//typedef Support *Supports;
typedef unsigned long ulong;


extern float epsilon;
extern Class nclasses;
extern Attribute nattributes;
extern std::map<int, int> attrFeat;
extern bool verbose;
extern int ncall;
extern float spectime;
extern float comptime;


#define NO_SUP INT_MAX // SHRT_MAX
#define NO_ERR FLT_MAX
#define NO_CACHE_LIMIT 0
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
#define forEachClass(n) for ( Class n = 0; n < nclasses; ++n )
// loop in each index in an array
#define forEach(index, array) for ( int index = 0; index < array.size; ++index )
// redefine a class name to make it short
#define FND Freq_NodeData*
// redefine a class name to make it short
#define FNDM Freq_NodeDataManager*


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
SupportClass sumSupports(Supports supports);

// return dest which is array of substraction of src2 from src1
void minSupports(Supports src1, Supports src2, Supports dest);

// return dest which is array of addition of src2 from src1
void plusSupports(Supports src1, Supports src2, Supports dest);

// return dest which is array of substraction of src2 from src1
void subSupports(Supports src1, Supports src2, Supports dest);

bool floatEqual(float f1, float f2);

inline void set_0(ulong &word, int n) {word &= ~(1UL << n);}

inline void set_1(ulong &word, int n) {word |= (1UL << n);}

int countSetBits(unsigned long i);


// the array is a light-weight vector that does not do copying or resizing of storage space.
template<class A>
struct Array {
public:

    A *elts; //the "array" of elements
    int size; //the real size of elements in the array while "allocsize" is the allocated size

    Array() {
        elts = nullptr;
        size = 0;
    }

    Array(A *elts_, int size_) {
        elts = elts_;
        size = size_;
    }

    /*Array(const Array<A> &ar) {
        size = ar.size;
        elts = new A[ar.size];
        forEach(i, ar) elts[i] = ar.elts[i];
    }*/

    Array(int allocsize, int size_) {
        size = size_;
        elts = new A[allocsize];
    }

    void duplicate(const Array<A> ar) {
//        if (elts) delete[] elts;
        size = ar.size;
        elts = new A[ar.size];
        forEach(i, ar) elts[i] = ar.elts[i];
    }

    Array<A> duplicate() {
        Array<A> newArr(size, size);
        for (int i = 0; i < size; ++i) newArr[i] = elts[i];
        return newArr;
    }

    Array<A>& operator=(const Array<A>& ar) = default;

    /*Array<A>& operator=(const Array<A>& ar) {
        elts = ar.elts;
        size = ar.size;
        return *this;
    }*/

    //~Array() {} //destructor does not do anything. Make sure you call free method after using the object

    void alloc(int size_) {
        size = size_;
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

    int getSize() {
        return size;
    }

    A &operator[](int i) { return elts[i]; }

    bool operator==(const Array<A> &rhs) const {
        if (elts == rhs.elts) return true;
        if (size != rhs.size) return false;
        forEach(i, rhs) if (elts[i] != *(rhs.elts + i)) return false;
        return true;
    }

    class iterator {
    public:
        iterator(A *ptr) : ptr(ptr) {}

        iterator operator++() {
            ++ptr;
            return *this;
        }

        bool operator!=(const iterator &other) const { return ptr != other.ptr; }

        // const is added in front of signature to only allow read. It is much faster but we lose in write
        const A &operator*() const { return *ptr; }

    private:
        A *ptr;
    };

    iterator begin() const { return iterator(elts); }

    iterator end() const { return iterator(elts + size); }
};

/*// custom hash can be a standalone function object:
struct MyHash {
    std::size_t operator()(S const &s) const noexcept {
        std::size_t h1 = std::hash<std::string>{}(s.first_name);
        std::size_t h2 = std::hash<std::string>{}(s.last_name);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }
};*/

// custom specialization of std::hash can be injected in namespace std
namespace std {
    template<class A>
    struct hash<Array<A>> {
        std::size_t operator()(const Array<A>& array) const noexcept {
            std::size_t h = array.size;
            for (auto elt: array) h ^= elt + 0x9e3779b9 + 64 * h + h / 4;
            return h;
        }
    };
}


void merge(Array<Item> src1, Array<Item> src2, Array<Item> dest);

void addItem(Array<Item> src1, Item item, Array<Item> dest);

Array<Item> addItem(Array<Item> src1, Item item);

void printItemset(Array<Item> itemset, bool force = false);


#endif