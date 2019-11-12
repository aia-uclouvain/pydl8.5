#include "dataContinuous.h"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>
#include <math.h>


using namespace std;

DataContinuous::DataContinuous(bool save):save(save) {

}


DataContinuous::~DataContinuous() {

}

void findAndReplaceAll(std::string & data, std::string toSearch, std::string replaceStr)
{
  // Get the first occurrence
  size_t pos = data.find(toSearch);

  // Repeat till end is reached
  while( pos != std::string::npos)
  {
    // Replace this occurrence of Sub String
    data.replace(pos, toSearch.size(), replaceStr);
    // Get the next occurrence from the current position
    pos =data.find(toSearch, pos + replaceStr.size());
  }
}

template <typename A, typename B>
void zip(std::vector<A> a, std::vector<B> b, std::vector<std::pair<A,B>> &zipped) {
  for(size_t i=0; i<a.size(); ++i) {
    zipped.push_back(std::make_pair(a[i], b[i]));
  }
}

template <typename A, typename B>
void unzip(const std::vector<std::pair<A, B>> &zipped, float* a, int* b) {
  for(size_t i=0; i<zipped.size(); i++) {
    a[i] = zipped[i].first;
    b[i] = zipped[i].second;
  }
}

std::vector<std::vector<int> > transpose(std::vector<std::vector<int> > data) {
  std::vector<std::vector<int> > result;

  for (int i = 0; i < data[0].size(); ++i) {
    std::vector<int> line;
    for (int j = 0; j < data.size(); ++j) {
      line.push_back(data[j][i]);
    }
    result.push_back(line);
  }

  return result;
}

/*!
    \fn Data::read ( const char *filename )
 */
void DataContinuous::read ( const char *filename ) {

  fstream fin;
  fin.open(filename, ios::in);

  vector<vector<float>> data;
  string line, word;

  getline(fin, line);
  stringstream sline(line);
  while (getline(sline, word, ';')) {
    colNames.push_back(word);
  }

  int nColumns = count(line.begin(), line.end(), ';');
  for (int k = 0; k < nColumns; ++k) {
    data.push_back(vector<float>());
  }

  while (getline(fin, line)) {
    stringstream s(line);

    int i = 0;
    while (getline(s, word, ';')) {
      if (i == nColumns)
        c.push_back(std::stoi(word));
      else
        data[i].push_back(std::stof(word));
      ++i;
    }
  }
  ntransactions = data[0].size();
  //int nContattributes = data.size();
  nclasses = *max_element(c.begin(),c.end()) + 1;
  ::nclasses = nclasses;

  supports = zeroSupports();

  fin.close();

  binarize(data);

  if (save){
      string out(filename);
      findAndReplaceAll(out, "continuous", "generated/csv");
      write_binary(out);
      cout << out << endl;
      findAndReplaceAll(out, "generated/csv", "generated/dl85");
      write_binary_dl8(out);
      cout << out << endl;
  }


}


void DataContinuous::binarize(vector<vector<float>> toBinarize) {

  int endAttr = toBinarize.size();
  /*int nFeatures_before =  endAttr;
  int nFeatures_after =  0;
  int nFeatures_added = 0;*/

  for (int i = 0; i < endAttr; i++) {
    //initialize vector to store breaks
    vector<float> breaks = vector<float>();

    /*%%%%%%%%% get feature vector and target vector ordered by feature %%%%%%%%%%%%*/
    std::vector<std::pair<float,int>> zipped;
    zip(toBinarize[i], c, zipped);
    std::sort(std::begin(zipped), std::end(zipped),
              [&](const pair<float,int> & a, const pair<float,int> & b)
              {
                  return a.first < b.first;
              });

    auto * target = new int[ntransactions];
    auto * col = new float [ntransactions];
    unzip(zipped, col, target);
    /*%%%%%%%%%%%%%%% end %%%%%%%%%%%*/

    float val = col[0];
    float classVal = target[0];
    bool diffVal = false;

    if (i == 0)
      ++supports[int(target[0])];

    for (int j = 1; j < ntransactions; j++) {

      if (i == 0)
        ++supports[int(target[j])];

      if(col[j] != val){//la valeur du feature a changé

        if(target[j] != classVal){//la valeur de la classe a changé
          breaks.push_back(col[j-1]);
          diffVal = false;
        }
        else{ //la valeur de la classe n'a pas changé

          if (diffVal){//la valeur de la classe a changé la dernière fois
            breaks.push_back(col[j-1]);
            diffVal = false;
          }
          else{ // la valeur de la classe n'a pas changé la dernière fois
            double tmpTarget = classVal;
            double tmpVal = col[j];
            bool change = false;
            for (int k = j+1; k < ntransactions; k++) {
              if(col[k] != tmpVal)
                break;
              if(target[k] != tmpTarget){
                change = true;
                break;
              }
            }
            if(change){
              breaks.push_back(col[j-1]);

              diffVal = false;
            }
            else{
              val = col[j];
              continue;
            }

          }
        }
      }
      else {
        if(target[j] != classVal){
          diffVal = true;
        }
      }
      val = col[j];
      classVal = target[j];
    }


    vector<vector<int>> newCols;

    /*for (int l = 0; l < breaks.size(); ++l) {
      vector<int> tmpCol;
      for (int k = 0; k < ntransactions; k++) {
        if(toBinarize[i][k] <= breaks[l])
          tmpCol.push_back(1);
        else
          tmpCol.push_back(0);
      }
      newCols.push_back(tmpCol);
      string minBound = "'(-inf";
      string maxBound = "" + to_string(breaks[l]) + "]'";
      //string maxBound = "" + to_string( ( breaks[j] + breaks[j+1] ) / 2 ) + "]'";

      //string name = colNames[i] + "=" + minBound + "-" + maxBound;
      //names.push_back(colNames[i] + "=" + minBound + "-" + maxBound);
      string str = to_string(roundf( (breaks[l] + breaks[l+1]) / 2 * 100) / 100);
      str.erase ( str.find('.') + 3, std::string::npos );
      names.push_back(colNames[i] + " <= " + str);
    }*/

    for (float j : breaks) {
      vector<int> tmpCol;
      for (int k = 0; k < ntransactions; k++) {
        if(toBinarize[i][k] <= j)
          tmpCol.push_back(1);
        else
          tmpCol.push_back(0);
      }
      newCols.push_back(tmpCol);

      //string minBound = "'(-inf";
      //string maxBound = "" + to_string(j) + "]'";
      //string maxBound = "" + to_string( ( breaks[j] + breaks[j+1] ) / 2 ) + "]'";
      //string name = colNames[i] + "=" + minBound + "-" + maxBound;
      //names.push_back(colNames[i] + "=" + minBound + "-" + maxBound);
      string str = to_string(roundf(j * 100) / 100);
      str.erase ( str.find('.') + 3, std::string::npos );
      names.push_back(colNames[i] + " <= " + str);
    }

    int start = int(b.size());
    for (const auto &newCol : newCols) {
      b.push_back(newCol);
    }
    int stop = int(b.size());

    for (int l = start; l < stop; ++l) {
      attrFeat[l] = i;
    }

  }
  b = transpose(b);
  nattributes = int(b[0].size());
  ::nattributes = nattributes;

  /*nFeatures_after = b[0].size();
  nFeatures_added = nFeatures_after - nFeatures_before;
  cout << "before = " << nFeatures_before << " after = " << nFeatures_after << " added = " << nFeatures_added << " names = " << names.size() << endl;*/

}

void DataContinuous::write_binary(string filename){
  /*%%%%%%%%%%%%%%%%% write discretized dataset in a file %%%%%%%%%%%%%%%%%*/
  fstream fout;

  // opens an existing csv file or creates a new file.
  fout.open(filename, ios::out);

  for (int m = 0; m < names.size(); ++m) {
    fout << names[m] << ";";
  }
  fout << "target" << '\n';

  // Read the input_14
  for (int i = 0; i < b.size(); i++) {

    for (int j = 0; j < b[0].size(); ++j) {

      fout << b[i][j] << ";";

    }
    fout << c[i] << "\n";
  }

  fout.close();
  /*%%%%%%%%%%%%% end writing %%%%%%%%%%%%%*/
}

void DataContinuous::write_binary_dl8(string filename){
  /*%%%%%%%%%%%%%%%%% write discretized dataset in a file %%%%%%%%%%%%%%%%%*/
  fstream fout;

  // opens an existing csv file or creates a new file.
  fout.open(filename, ios::out);


  // Read the input_14
  for (int i = 0; i < b.size(); i++) {

    fout << c[i];

    for (int j = 0; j < b[0].size(); ++j) {

      fout << " " << b[i][j];

    }
    fout << "\n";
  }

  fout.close();
  /*%%%%%%%%%%%%% end writing %%%%%%%%%%%%%*/
}




