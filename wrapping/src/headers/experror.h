#ifndef EXPERROR_H
#define EXPERROR_H
#include "globals.h"

enum ExpErrorType { Zero, C45, J48, Half };


class ExpError{
public:
    ExpError();

    virtual ~ExpError();
    virtual ExpErrorType getExpErrorType () = 0;
    virtual Error addError ( Support total, Error error, Support datasize ) = 0;
};

class ExpError_Zero:public ExpError {
public:
    ExpError_Zero ():ExpError () { }
    
    ExpErrorType getExpErrorType () { return Zero; }
    Error addError ( Support total, Error error, Support datasize ) { return 0.0; }
};

class ExpError_C45:public ExpError {
public:
    ExpError_C45 ( float CF );
    
    ExpErrorType getExpErrorType () { return C45; }
    Error addError ( Support total, Error error, Support datasize );
  private:
    float Coeff; // computed from CF
    float CF; // confidence factor
};

class ExpError_J48:public ExpError {
public:
    ExpError_J48 ( float CF );
    
    ExpErrorType getExpErrorType () { return J48; }
    Error addError ( Support total, Error error, Support datasize );
private:
    float normalInverse ( double CF );
    double p1evl( double x, double coef[], int N );
    double polevl( double x, double coef[], int N );    
    float CF; // confidence factor
    double z; // computed from CF
};

class ExpError_Half:public ExpError {
public:
    ExpError_Half ():ExpError () {} 
    
    ExpErrorType getExpErrorType () { return Half; }
    Error addError ( Support total, Error error, Support datasize ){ return 0.5; }
};

#endif
