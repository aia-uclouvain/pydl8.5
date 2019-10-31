
#include <iostream>
#include <stdio.h>
#include "data.h"

using namespace std;

Data::Data() {
  
}


Data::~Data() {
  /*delete[] b[0];
  delete[] b;
  delete[] c;*/
}


/*
char getint ( FILE *in, int &item ) {
  char c = getc ( in );
  item = 0;
  while ( c == ' ' || c == '\n' || c == '\t' || c == '\10') 
    c = getc ( in );
  while ( ( c >= '0' ) && ( c <= '9' ) ) {
    item *=10;
    item += int(c)-int('0');
    c = getc(in);
  }
  return c;
}
*/

/*!
    \fn Data::read ( const char *filename )
 */
/*void Data::read ( const char *filename ) {
  FILE *in = fopen ( filename, "rt" );
  int val;
  char a;
  Class classnum;
  nattributes = 0;

  // -------- first we read the file to obtain the numbers of classes, etc.

  /// get number of features
  a = getint ( in, nclasses );
  while ( a != '\n' && !feof ( in ) ) {
    a = getint ( in, val );
    ++nattributes;
  }

  /// get number of transactions
  ntransactions = 1;
  
  while ( !feof ( in) ) {
    getint ( in, classnum );
    for ( int i = 0; i < nattributes; ++i ) 
      getint ( in, val );
    if ( classnum > nclasses )
      nclasses = classnum;
    ++ntransactions;
  }
  
  // minor corrections
  ++nclasses;
  --ntransactions;
  fclose ( in );
  
  ::nattributes = nattributes;
  ::nclasses = nclasses;
  
  // Now we allocate the matrix with data
  b = new Bool*[ntransactions];
  // group allocations together will increase speed (hopefully) as all
  // data is close to each other in memory
  b[0] = new Bool[ntransactions * nattributes]; 
  c = new Class[ntransactions];
  supports = zeroSupports();
  
  in = fopen ( filename, "rt" );
  for ( int i = 0; i < ntransactions; ++i ) {
    getint ( in, val ); 
    c[i] = val;
    ++supports[val];
    b[i] = b[0] + ( i * nattributes );
    for ( int j = 0; j < nattributes; ++j ) {
      getint ( in, val );
      b[i][j] = val % 2;
    }
  }
  fclose ( in ); 
} */


