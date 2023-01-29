#include "globals.h"
#include <cmath>


GlobalParams* GlobalParams::instance = nullptr;

ErrorVals newErrorVals () {
  return new ErrorVal[GlobalParams::getInstance()->nclasses];
}

ErrorVals zeroErrorVals () {
  ErrorVals supports = newErrorVals();
    zeroErrorVals(supports);
  return supports;
}

void zeroErrorVals (ErrorVals supports ) {
  forEachClass ( i ) 
    supports[i] = 0;
}

//void deleteErrorVals (constErrorVals supports ) {
void deleteErrorVals (ErrorVals supports ) {
  delete[] supports;
}

void copyErrorVals (constErrorVals src, ErrorVals dest ) {
  forEachClass ( i ) dest[i] = src[i];
}

ErrorVals copyErrorVals (ErrorVals supports ) {
  ErrorVals supports2 = newErrorVals();
  copyErrorVals(supports, supports2);
  return supports2;
}

ErrorVal sumErrorVals (constErrorVals supports ) {
  ErrorVal sum = 0;
  forEachClass ( i ) sum += supports[i];
  return sum;
}

void addErrorVals (constErrorVals src1, constErrorVals src2, ErrorVals dest ) {
  forEachClass ( i ) dest[i] = src1[i] + src2[i];
}

void subErrorVals (constErrorVals src1, constErrorVals src2, ErrorVals dest ) {
    forEachClass ( i ) dest[i] = src1[i] - src2[i];
}

ErrorVals subErrorVals (constErrorVals src1, constErrorVals src2 ) {
    ErrorVals sub = zeroErrorVals();
    forEachClass ( i ) sub[i] = src1[i] - src2[i];
    return sub;
}

int countSetBits(unsigned long i) {
    i = i - ((i >> 1) & 0x5555555555555555UL);
    i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
    return (int)((((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56);
}

void merge ( const Itemset &src1, const Itemset &src2, Itemset &dest ) {
  int i = 0, j = 0, k = 0;
  while ( i < src1.size() && j < src2.size() ) {
      if (src1[i] < src2[j]) dest[k++] = src1[i++];
      else dest[k++] = src2[j++];
  }
  while ( i < src1.size() ) dest[k++] = src1[i++];
  while ( j < src2.size() ) dest[k++] = src2[j++];
}

void addItem (const Itemset &src, Item item, Itemset &dest ) {
  int i = 0, j = 0, k = 0;
  while (i < src.size() && j < 1 ) {
      if (src[i] < item) dest[k++] = src[i++];
      else { dest[k++] = item; j++; }
  }
  while (i < src.size() ) dest[k++] = src[i++];
  if ( j < 1 ) dest[k++] = item;
}

Itemset addItem (const Itemset &src, Item item, bool quiet) {
    Itemset dest(src.size() + 1);
    int i = 0, j = 0, k = 0;
    while (i < src.size() && j < 1 ) {
        if (src[i] < item ) dest[k++] = src[i++];
        else { dest[k++] = item; j++; }
    }
    while (i < src.size() ) dest[k++] = src[i++];
    if ( j < 1 ) dest[k++] = item;

    if (not quiet and GlobalParams::getInstance()->verbose) {
        std::cout << "-\nitemset avant ajout : "; printItemset(src);
        std::cout << "Item à ajouter : " << item << std::endl;
        std::cout << "itemset après ajout : "; printItemset(dest);
    }
    return dest;
}

void printItemset(const Itemset &itemset, bool force, bool newline) {
    if (GlobalParams::getInstance()->verbose or force) {
        if (itemset.empty()) std::cout << "\\phi";
        for (const auto& item : itemset) std::cout << item << ",";
        if (newline) std::cout << std::endl;
    }
}

bool floatEqual(float f1, float f2) {
    return fabs(f1 - f2) <= FLT_EPSILON;
}

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

/*bool floatEqual(float f1, float f2)
{
    if (abs(f1 - f2) <= epsilon)
        return true;
    return abs(f1 - f2) <= epsilon * std::max(abs(f1), abs(f2));
}*/

