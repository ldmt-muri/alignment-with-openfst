#ifndef _VOCAB_ENCODER_H_
#define _VOCAB_ENCODER_H_

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <assert.h>

#include "StringUtils.h"

class VocabDecoder {
 public:
 VocabDecoder(const std::string& vocabFilename) {
    UNK = "_unk_";
    std::ifstream vocabFile(vocabFilename.c_str(), std::ios::in);
    std::string line;
    while(getline(vocabFile, line)) {
      if(line.size() == 0) {
	continue;
      }
      std::vector<string> splits;
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

