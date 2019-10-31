#ifndef DATABINARY_H
#define DATABINARY_H
#include "globals.h"
#include "data.h"
#include <iostream>


class DataBinary : public Data{
public:
    DataBinary();

    ~DataBinary();
    /// read input file, fill variables nattributes, ntransactions, nclasses, c (array of target), b (array of array of dataset).
    /// Initialize supports array for each class with value 0
    void read ( const char *filename ) override;
    /// check if attribute is selected in the transaction ==> return true or false
    inline Bool isIn ( Transaction transaction, Attribute attr ) const { if (transaction < 0) std::cout << "\t\t\t\t\t<<<<transaction negative : " << transaction; return b[transaction][attr]; }
    /// return class(target) of transaction
    inline Class targetClass ( Transaction transaction ) const { return c[transaction]; }
    /// get number of transactions
    //int getNTransactions () const { return ntransactions; }
    /// get number of features
    //int getNAttributes () const { return nattributes; }
    /// get number of transactions
    //int getNClasses () const { return nclasses; }
    /// get array of support of each class
    //Supports getSupports () const { return supports; }
private:
    //Supports supports; /// array of support for each class
    Bool **b; /// matrix of data
    Class *c; /// vector of target
    //Transaction ntransactions; /// number of transactions
    //Attribute nattributes; /// number of features
    //Class nclasses; /// number of classes
};

#endif
