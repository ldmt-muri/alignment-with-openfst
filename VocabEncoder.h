#ifndef _VOCAB_ENCODER_H_
#define _VOCAB_ENCODER_H_

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include <assert.h>
#include <limits.h>

#include "StringUtils.h"

using namespace std;

class VocabEncoder {
 public:
  int nextId;
  map<string, int> tokenToInt;
  map<int, string> intToToken;
  std::string UNK;
  bool useUnk;
  std::set<int> closedVocab;
  
 public:
  VocabEncoder() {
    nextId = 2;
    UNK = "_unk_";

    // encode unk
    tokenToInt[UNK] = nextId;
    intToToken[nextId++] = UNK;
  }

  VocabEncoder(const std::string& textFilename) {
    useUnk = true;
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
    useUnk = true;
  }

  bool IsClosedVocab(int wordId) const {
    return (closedVocab.find(wordId) != closedVocab.end());
  }

  void AddToClosedVocab(std::string &word) {
    int code = Encode(word, false);
    closedVocab.insert(code);
  }

  int Encode(const string& token, bool explicitUseUnk) {
    if(tokenToInt.count(token) == 0) {
      if(explicitUseUnk) {
	return tokenToInt[UNK];
      }	else {
	tokenToInt[token] = nextId;
	intToToken[nextId++] = token;
	assert(nextId != INT_MAX);
	return tokenToInt[token];
      }
    } else {
      return tokenToInt[token];
    }
  }

  int Encode(const string& token) {
    return Encode(token, useUnk);
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
  
  // read each line in the text file, encodes each sentence into vector<int> and appends it into 'data'
  // assumptions: data is empty
  void Read(const std::string &textFilename, vector<vector<int> > &data) {
    
    assert(data.size() == 0);
    
    // open data file
    std::ifstream textFile(textFilename.c_str(), std::ios::in);
    
    // for each line
    std::string line;
    int lineNumber = -1;
    while(getline(textFile, line)) {
      
      // skip empty lines
      if(line.size() == 0) {
	continue;
      }
      lineNumber++;
      
      // split tokens
      std::vector<string> splits;
      StringUtils::SplitString(line, ' ', splits);
      
      // encode tokens    
      data.resize(lineNumber+1);
      Encode(splits, data[lineNumber]);
    }
  }
  
  void PersistVocab(string filename) {
    std::ofstream vocabFile(filename.c_str(), std::ios::out);
    for(map<int, string>::const_iterator intToTokenIter = intToToken.begin(); intToTokenIter != intToToken.end(); intToTokenIter++) {
      bool inClosedVocab = closedVocab.find(intToTokenIter->first) != closedVocab.end();
      // c for closed, o for open
      vocabFile << intToTokenIter->first << " " << intToTokenIter->second << " " << (inClosedVocab? "c" : "o") << endl;
    }
    vocabFile.close();
  }

  const std::string& Decode(int wordId) const {
    if(intToToken.count(wordId) == 0) {
      return this->UNK;
    } else {
      return intToToken.find(wordId)->second;
    }
  }

};

class VocabDecoder {
 public:
  VocabDecoder(const VocabDecoder& another) {
    vocab = another.vocab;
    UNK = another.UNK;
  }

  VocabDecoder(VocabDecoder& another) {
    vocab = another.vocab;
    UNK = another.UNK;
  }

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
      vocab[wordId].assign(splits[1]);
      if(splits[2] == "c") {
	closedVocab.insert(wordId);
      } else if(splits[2] == "o") {
	// do nothing
      } else {
	// format error!
	assert(false);
      }
      
      //      cerr << "vocab[" << wordId << "] = " << vocab.find(wordId)->second << endl;
      //      cerr << "&(splits[0])=" << &(splits[0]) << ", &(vocab.find(wordId)->second)=" << &(vocab.find(wordId)->second) << endl;
    }
    vocab[1] = "_null_";
    vocab[-1] = "_<s>_";
    //vocab[0] = "_zero_"; // shouldn't happen!
  }
  
  const std::string& Decode(int wordId) const {
    if(vocab.find(wordId) == vocab.end()) {
      return this->UNK;
    } else {
      return vocab.find(wordId)->second;
    }
  }

  bool IsClosedVocab(int wordId) const {
    return (closedVocab.find(wordId) != closedVocab.end());
  }

  std::map<int, std::string> vocab;
  std::string UNK;
  std::set<int> closedVocab;
};

#endif

