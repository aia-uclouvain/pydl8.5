#ifndef DATACONTINUOUS_H
#define DATACONTINUOUS_H
#include "globals.h"
#include "data.h"
#include <iostream>
#include <vector>


class DataContinuous : public Data{
public:
    DataContinuous(bool save);

    ~DataContinuous();
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

    void binarize(std::vector<std::vector<float>> toBinarize);
    void write_binary(std::string filename);
    void write_binary_dl8(std::string filename);
    std::vector<std::string> names;

private:
    std::vector<std::vector<int>> b; /// matrix of data
    std::vector<int> c; /// vector of target
    std::vector<std::string> colNames;
    bool save;
};

#endif
