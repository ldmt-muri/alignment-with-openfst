#ifndef _STRING_UTILS_H_
#define _STRING_UTILS_H_

#include <string>
#include <vector>
#include <sstream>
#include <set>

typedef std::string string;
typedef std::stringstream stringstream;

class StringUtils {
 public:
  // string split
  static void SplitString(const string& s, char delim, std::vector<string>& elems) {
    stringstream ss(s);
    string item;
    while(getline(ss, item, delim)) {
      elems.push_back(item);
    }
  }

  // read int tokens
  static void ReadIntTokens(const string& sentence, std::vector<int>& intTokens) {
    std::vector<string> stringTokens;
    SplitString(sentence, ' ', stringTokens);
    for (std::vector<string>::iterator tokensIter = stringTokens.begin(); 
	 tokensIter < stringTokens.end(); tokensIter++) {
      int intToken;
      stringstream stringToken(*tokensIter);
      stringToken >> intToken;
      intTokens.push_back(intToken);
    }
  }

  // read int token set
  static void ReadIntTokens(const string& sentence, std::set<int>& intTokens) {
    std::vector<string> stringTokens;
    SplitString(sentence, ' ', stringTokens);
    for (std::vector<string>::iterator tokensIter = stringTokens.begin(); 
	 tokensIter < stringTokens.end(); tokensIter++) {
      int intToken;
      stringstream stringToken(*tokensIter);
      stringToken >> intToken;
      intTokens.insert(intToken);
    }
  }

  // compute a measure of orthographic similarity between two words
  static double ComputeOrthographicSimilarity(const std::string& srcWord, const std::string& tgtWord) {
    if(srcWord.length() == 0 || tgtWord.length() == 0) {
      return 0.0;
    }
    int levenshteinDistance = LevenshteinDistance(srcWord, tgtWord);
    if(levenshteinDistance > (srcWord.length() + tgtWord.length()) / 2) {
      return 0.0;
    } else {
      double similarity = (srcWord.length() + tgtWord.length()) / ((double)levenshteinDistance + 1);
      return similarity;
    }
  }
  
  // levenshtein distance
  static int LevenshteinDistance(const std::string& x, const std::string& y) {
    if(x.length() == 0 && y.length() == 0) {
      return 0;
    }
    
    if(x.length() == 0) {
      return y.length();
    } else if (y.length() == 0) {
      return x.length();
    } else {
      int cost = x[0] != y[0]? 1 : 0;
      std::string xSuffix = x.substr(1);
    std::string ySuffix = y.substr(1);
    return std::min( std::min( LevenshteinDistance(xSuffix, y) + 1,
			       LevenshteinDistance(x, ySuffix) + 1),
		     LevenshteinDistance(xSuffix, ySuffix) + cost);
    }
  }


};

#endif
