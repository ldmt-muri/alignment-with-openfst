#ifndef _STRING_UTILS_H_
#define _STRING_UTILS_H_

#include <string>
#include <sstream>
#include <set>
#include <vector>

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

};

#endif
