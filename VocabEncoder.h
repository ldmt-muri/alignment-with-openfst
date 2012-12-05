#ifndef _VOCAB_ENCODER_H_
#define _VOCAB_ENCODER_H_

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <assert.h>
#include <limits.h>

#include "StringUtils.h"

using namespace std;

class VocabEncoder {
 public:
  VocabEncoder(const std::string& textFilename) {
    nextId = 2;
    UNK = "_unk_";

    // encode unk
    tokenToInt[UNK] = nextId;
    intToToken[nextId++] = UNK;

    // create token-int correspondnence
    std::ifstream textFile(textFilename.c_str(), std::ios::in);
    std::string line;
    while(getline(textFile, line)) {
      if(line.size() == 0) {
	continue;
      }
      std::vector<string> splits;
      StringUtils::SplitString(line, ' ', splits);
      for(std::vector<string>::const_iterator tokenIter = splits.begin(); 
	  tokenIter != splits.end();
	  tokenIter++) {
	if(tokenToInt.count(*tokenIter) == 0) {
	  tokenToInt[*tokenIter] = nextId;
	  intToToken[nextId++] = *tokenIter;
	  assert(nextId != INT_MAX);
	}
      }
    }
  }

  int Encode(const string& token) {
    if(tokenToInt.count(token) == 0) {
      return tokenToInt[UNK];
    } else {
      return tokenToInt[token];
    }
  }

  void Encode(const std::vector<std::string>& tokens, vector<int>& ids) {
    assert(ids.size() == 0);
    for(vector<string>::const_iterator tokenIter = tokens.begin();
	tokenIter != tokens.end();
	tokenIter++) {
      ids.push_back(Encode(*tokenIter));
    }
    assert(ids.size() == tokens.size());
  }

  std::map<int, std::string> intToToken;
  std::map<std::string, int> tokenToInt;
  int nextId;
  std::string UNK;
};

class VocabDecoder {
 public:
 VocabDecoder(const std::string& vocabFilename) {
    std::ifstream vocabFile(vocabFilename.c_str(), std::ios::in);
    std::string line;
    UNK = "_unk_";
    while(getline(vocabFile, line)) {
      if(line.size() == 0) {
	continue;
      }
      std::vector<std::string> splits;
      StringUtils::SplitString(line, ' ', splits);
      stringstream ss(splits[0]);
      int wordId;
      ss >> wordId;
      vocab[wordId] = splits[1];
    }
    vocab[1] = "_null_";
    vocab[-1] = "_<s>_";
    //vocab[0] = "_zero_"; // shouldn't happen!
  }

  const std::string& Decode(int wordId) const {
    if(vocab.count(wordId) == 0) {
      return this->UNK;
    } else {
      return vocab.find(wordId)->second;
    }
  }

  std::map<int, std::string> vocab;
  std::string UNK;
};

#endif

