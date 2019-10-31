#ifndef DATA_H
#define DATA_H
#include "globals.h"
#include <iostream>


class Data{
public:
    Data();

    ~Data();
    /// read input file, fill variables nattributes, ntransactions, nclasses, c (array of target), b (array of array of dataset).
    /// Initialize supports array for each class with value 0
    virtual void read ( const char *filename ) = 0;
    /// check if attribute is selected in the transaction ==> return true or false
    virtual Bool isIn ( Transaction transaction, Attribute attr ) const = 0;// const { if (transaction < 0) std::cout << "\t\t\t\t\t<<<<transaction negative : " << transaction; return b[transaction][attr]; }
    /// return class(target) of transaction
    virtual Class targetClass ( Transaction transaction ) const = 0;
    /// get number of transactions
    virtual int getNTransactions () const { return ntransactions; }
    /// get number of features
    virtual int getNAttributes () const { return nattributes; }
    /// get number of transactions
    int getNClasses () const { return nclasses; }
    /// get array of support of each class
    Supports getSupports () const { return supports; }

protected:
    Supports supports; /// array of support for each class
    Transaction ntransactions; /// number of transactions
    Attribute nattributes; /// number of features
    Class nclasses; /// number of classes
};

#endif
