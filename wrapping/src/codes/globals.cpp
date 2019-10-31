#include "globals.h"
#include <math.h>

Class nclasses;
Attribute nattributes;
std::map<int,int> attrFeat;

Supports newSupports () {
  return new Support[nclasses];
}

Supports zeroSupports () {
  Supports supports = newSupports();
  zeroSupports ( supports );
  return supports;
}

void zeroSupports ( Supports supports ) {
  forEachClass ( i ) 
    supports[i] = 0;
}

void deleteSupports ( Supports supports ) {
  delete[] supports;
}

void copySupports ( Supports src, Supports dest ) {
  forEachClass ( i )
    dest[i] = src[i];
}

Supports copySupports ( Supports supports ) {
  Supports supports2 = newSupports ();
  copySupports ( supports, supports2 );
  return supports2;
}

Support sumSupports ( Supports supports ) {
  Support sum = 0;
  forEachClass ( i )
    sum += supports[i];
  return sum;
}

void minSupports ( Supports src1, Supports src2, Supports dest ) {
  forEachClass ( i )
    dest[i] = src1[i] - src2[i];
}

void plusSupports ( Supports src1, Supports src2, Supports dest ) {
  forEachClass ( i )
    dest[i] = src1[i] + src2[i];
}

void merge ( Array<Item> src1, Array<Item> src2, Array<Item> dest ) {
  int i = 0, j = 0, k = 0;
  while ( i < src1.size && j < src2.size )
    if ( src1[i] < src2[j] )
      dest[k++] = src1[i++];
    else
      dest[k++] = src2[j++];
  while ( i < src1.size )
    dest[k++] = src1[i++];
  while ( j < src2.size )
    dest[k++] = src2[j++];
}

void addItem ( Array<Item> src1, Item item, Array<Item> dest ) {
  int i = 0, j = 0, k = 0;
  while ( i < src1.size && j < 1 )
    if ( src1[i] < item )
      dest[k++] = src1[i++];
    else{
      dest[k++] = item;
      j++;
    }
  while ( i < src1.size )
    dest[k++] = src1[i++];
  if ( j < 1 )
    dest[k++] = item;
}

