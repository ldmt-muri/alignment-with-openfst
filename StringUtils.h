#ifndef _STRING_UTILS_H_
#define _STRING_UTILS_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
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

  static std::string IntVectorToString(const std::vector<int> &numbers) {
    std::stringstream ss("");
    for(int i = 0; i < numbers.size(); i++) {
      ss << numbers[i] << " ";
    }
    return ss.str();
  }

  // 'tokens' is a vector of sentences
  // each sentence is a vector of space-separated tokens
  static void ReadTokens(const std::string &filename, std::vector<std::vector<std::string> > &tokens) {
    std::ifstream textFile(filename.c_str(), std::ios::in);
    std::string line;
    // for each line
    while(getline(textFile, line)) {
      // skip empty lines
      if(line.size() == 0) {
	continue;
      }
      std::vector<string> splits;
      SplitString(line, ' ', splits);
      tokens.push_back(splits);
    }
    textFile.close();
  }

  // 'tokens' is a vector of sentences
  // each sentence is a vector of space-separated tokens
  static void ReadTokens(const std::string &filename, std::vector<std::vector<int> > &tokens) {
    std::ifstream textFile(filename.c_str(), std::ios::in);
    std::string line;
    // for each line
    while(getline(textFile, line)) {
      // skip empty lines
      if(line.size() == 0) {
	continue;
      }
      std::vector<int> splits;
      ReadIntTokens(line, splits);
      tokens.push_back(splits);
    }
    textFile.close();
  }

  static void WriteTokens(const std::string &filename, std::vector<std::vector<int> > &tokens) {
    std::ofstream textFile(filename.c_str(), std::ios::out);
    for(int i = 0 ; i < tokens.size(); i++) {
      if(tokens[i].size() != 0) {
	textFile << tokens[i][0];
      }
      for(int j = 1 ; j < tokens[i].size(); j++) {
	textFile << " " << tokens[i][j];
      }
      textFile << "\n";
    }
    textFile.close();
  }
};

#endif
