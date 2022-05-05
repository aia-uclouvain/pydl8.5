#include "globals.h"
#include <math.h>

Class nclasses;
Attribute nattributes;
std::map<int,int> attrFeat;
int ncall = 0;
float spectime = 0;
float comptime = 0;
float epsilon = 1.0e-05f;
bool verbose = false;

Supports newSupports () {
  return new SupportClass[nclasses];
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

SupportClass sumSupports ( Supports supports ) {
  SupportClass sum = 0;
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

void subSupports ( Supports src1, Supports src2, Supports dest ) {
    forEachClass ( i )
        dest[i] = src1[i] - src2[i];
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

Array<Item> addItem ( Array<Item> src1, Item item ) {
    Array<Item> dest;
    dest.alloc(src1.size + 1);
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

    if (verbose) std::cout << "-\nitemset avant ajout : "; printItemset(src1);
    if (verbose) std::cout << "Item à ajouter : " << item << std::endl;
    if (verbose) std::cout << "itemset après ajout : "; printItemset(dest);
    return dest;
}

void printItemset(Array<Item> itemset) {
    if (verbose) {
        for (int i = 0; i < itemset.size; ++i) {
            std::cout << itemset[i] << ",";
        }
        std::cout << std::endl;
    }
}

bool floatEqual(float f1, float f2)
{
    return fabs(f1 - f2) <= FLT_EPSILON;
}

/*bool floatEqual(float f1, float f2)
{
    if (abs(f1 - f2) <= epsilon)
        return true;
    return abs(f1 - f2) <= epsilon * std::max(abs(f1), abs(f2));
}*/

int find_not_zero(std::string& str) {
    for (int i = 0; i < str.size(); ++i) {
        if (str.at(i) != '0') {
            if (str.at(i) == '.') return -1;
            else return i;
        }
    }
    return -1;
}

std::string custom_to_str(float val) {
    std::string valstr = std::to_string(val);
    if (valstr.at(valstr.size()-1) != '0') return valstr;
    std::reverse(valstr.begin(), valstr.end());
    auto index = find_not_zero(valstr);
    std::reverse(valstr.begin(), valstr.end());
    if (index == -1) return valstr.substr(0, valstr.find('.'));
    else {
        auto ind = valstr.size() - index - 1;
        return valstr.substr(0, ind + 1);
    }
}