//
// Created by Gael Aglin on 2019-10-08.
//

#ifndef DL85_DATABINARYPYTHON_H
#define DL85_DATABINARYPYTHON_H

#include "globals.h"
#include "data.h"
#include <iostream>


class DataBinaryPython : public Data {
public:
    DataBinaryPython(Supports supports, Transaction ntransactions, Attribute numattributes, Class nclasses, Bool *b, Class *c);

    ~DataBinaryPython();
    /// read input file, fill variables nattributes, ntransactions, nclasses, c (array of target), b (array of array of dataset).
    /// Initialize supports array for each class with value 0
    void read ( const char *filename ) override;
    /// check if attribute is selected in the transaction ==> return true or false
    inline Bool isIn ( Transaction transaction, Attribute attr ) const { if (transaction < 0) std::cout << "\t\t\t\t\t<<<<transaction negative : " << transaction; return b[transaction][attr]; }
    /// return class(target) of transaction
    inline Class targetClass ( Transaction transaction ) const { return c[transaction]; }
private:
    Bool **b; /// matrix of data
    Class *c; /// vector of target
};


#endif //DL85_DATABINARYPYTHON_H
