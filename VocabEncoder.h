#ifndef _VOCAB_ENCODER_H_
#define _VOCAB_ENCODER_H_

//#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include <boost/interprocess/detail/config_begin.hpp> 
#include <boost/interprocess/detail/workaround.hpp> 
#include <boost/interprocess/managed_shared_memory.hpp> 
#include <boost/interprocess/allocators/allocator.hpp> 
#include <boost/interprocess/containers/map.hpp> 
#include <boost/interprocess/containers/vector.hpp> 
#include <boost/interprocess/containers/string.hpp> 
#include <boost/unordered_map.hpp>
#include <assert.h>
#include <limits.h>

#include "LearningInfo.h"
#include "StringUtils.h"

using namespace std;

// special strings
typedef boost::interprocess::allocator<void, boost::interprocess::managed_shared_memory::segment_manager> 
  void_allocator; 
typedef boost::interprocess::allocator<char, boost::interprocess::managed_shared_memory::segment_manager> 
  char_allocator; 
typedef boost::interprocess::basic_string<char, std::char_traits<char>, char_allocator> 
  char_string; 

// Alias an STL compatible allocator of objects that need to be managed in the shared memory
typedef std::pair<const char_string, int64_t> 
  TokenToIntPair;
typedef const boost::interprocess::allocator<TokenToIntPair, boost::interprocess::managed_shared_memory::segment_manager> 
  ShmemTokenToIntPairAllocator;
typedef std::less<char_string> 
  StringLessThan;
typedef boost::interprocess::map<char_string, int64_t, StringLessThan, ShmemTokenToIntPairAllocator> 
  ShmemTokenToIntMap;

typedef std::pair<const int64_t, char_string> 
  IntToTokenPair;
typedef const boost::interprocess::allocator<IntToTokenPair, boost::interprocess::managed_shared_memory::segment_manager> 
  ShmemIntToTokenPairAllocator;
typedef boost::interprocess::map<int64_t, char_string, std::less<int64_t>, ShmemIntToTokenPairAllocator> 
  ShmemIntToTokenMap;

typedef boost::interprocess::allocator<char_string, boost::interprocess::managed_shared_memory::segment_manager> 
  ShmemStringAllocator;
typedef vector<char_string, ShmemStringAllocator> 
  ShmemStringVector;

class VocabEncoder {
 public:
  int64_t firstId, nextId;
  ShmemTokenToIntMap *tokenToInt;
  ShmemIntToTokenMap *intToToken;
  std::string UNK;
  char_string *UNK_char_string;
  std::set<int64_t> closedVocab;
  const LearningInfo &learningInfo;
  void_allocator *alloc_inst;

 public:

  void Init(unsigned firstId) {
    nextId = firstId;
    UNK = "_unk_";

    alloc_inst = new void_allocator(learningInfo.sharedMemorySegment->get_segment_manager());

    // create/find managed shared memory objects
    if(learningInfo.mpiWorld->rank() == 0) {
      // create
      intToToken = (ShmemIntToTokenMap *) MapToSharedMemory(true, "VocabEncoder::intToToken");
      tokenToInt = (ShmemTokenToIntMap *) MapToSharedMemory(true, "VocabEncoder::tokenToInt");
      // then sync
      bool dummy = false;
      boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, dummy, 0);
    } else {
      // sync
      bool dummy = false;
      boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, dummy, 0);
      // then find
      intToToken = (ShmemIntToTokenMap *) MapToSharedMemory(false, "VocabEncoder::intToToken");
      tokenToInt = (ShmemTokenToIntMap *) MapToSharedMemory(false, "VocabEncoder::tokenToInt");
    }

    // encode unk 
    UNK_char_string = new char_string(UNK.c_str(), *alloc_inst);
    tokenToInt->insert(TokenToIntPair(*UNK_char_string, nextId));
    intToToken->insert(IntToTokenPair(nextId++, *UNK_char_string));  
  }
  
 VocabEncoder(const LearningInfo &learningInfo, unsigned firstId = 2): learningInfo(learningInfo) {
    Init(firstId);
  }

 VocabEncoder(const std::string& textFilename, const LearningInfo &learningInfo, unsigned firstId = 2) : learningInfo(learningInfo) {
    Init(firstId);
    
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
        char_string token_char_string(tokenIter->c_str(), *alloc_inst);
        if((*tokenToInt).count(token_char_string) == 0) {
          tokenToInt->insert( TokenToIntPair(token_char_string, nextId) );
          intToToken->insert( IntToTokenPair(nextId++, token_char_string) );
          assert(nextId != LONG_MAX);
        }
      }
    }
  }

  void ReserveVocabSize(int64_t size) {
    return;
    // there is no way to do that with map implementation of intToToken and tokenToInt
    /*
    if((*tokenToInt).size() < size && intToToken->size() < size) {
      (*tokenToInt).rehash( ceil (1.0 * size / tokenToInt->max_load_factor()));
      intToToken->rehash( ceil (1.0 * size / intToToken->max_load_factor()));
      }*/
  }

  bool IsClosedVocab(int64_t wordId) const {
    return (closedVocab.find(wordId) != closedVocab.end());
  }

  void AddToClosedVocab(std::string &word) {
    int64_t code = ConstEncode(word);
    closedVocab.insert(code);
  }

  int64_t UnkInt() const {
    return (*tokenToInt).find(*UNK_char_string)->second;
  }

  string UnkString() {
    return UNK;
  }

  // a constant version of the encode function which guarantees that the underlying object state does not change
  // i.e. you cannot add new words to the vocab using this method
  int64_t ConstEncode(const string &token) const {
    char_string token_char_string(token.c_str(), *alloc_inst);
    if((*tokenToInt).count(token_char_string) == 0) {
      return (*tokenToInt).find(*UNK_char_string)->second;
    } else {
      return (*tokenToInt).find(token_char_string)->second;
    }    
  }

  int64_t Encode(const string& token) {
    char_string token_char_string(token.c_str(), *alloc_inst);
    if((*tokenToInt).count(token_char_string) == 0) {
      tokenToInt->insert(TokenToIntPair(token_char_string, nextId));
      intToToken->insert(IntToTokenPair(nextId++, token_char_string));
      assert(nextId != LONG_MAX);
      return (*tokenToInt)[token_char_string];
    } else {
      return (*tokenToInt)[token_char_string];
    }
  }
  
  void Encode(const std::vector<std::string>& tokens, vector<int64_t>& ids) {
    assert(ids.size() == 0);
    for(vector<string>::const_iterator tokenIter = tokens.begin();
        tokenIter != tokens.end();
        tokenIter++) {
      ids.push_back(Encode(*tokenIter));
    }
    assert(ids.size() == tokens.size());
  }
  
  void ReadParallelCorpus(const std::string &textFilename, vector<vector<int64_t> > &srcSents, vector<vector<int64_t> > &tgtSents) {
    ReadParallelCorpus(textFilename, srcSents, tgtSents, "", false);
  }
  
  void ReadParallelCorpus(const std::string &textFilename, vector<vector<int64_t> > &srcSents, vector<vector<int64_t> > &tgtSents, bool reverse) {
    ReadParallelCorpus(textFilename, srcSents, tgtSents, "", reverse);
  }
  
  // if nullToken is of length > 0, this token is inserted at position 0 for each src sentence.
  void ReadParallelCorpus(const std::string &textFilename, vector<vector<int64_t> > &srcSents, vector<vector<int64_t> > &tgtSents, const string &nullToken, bool reverse) {
    assert(srcSents.size() == 0 && tgtSents.size() == 0);
    
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
      srcSents.resize(lineNumber+1);
      tgtSents.resize(lineNumber+1);
      vector<int64_t> temp;
      Encode(splits, temp);
      assert(splits.size() == temp.size());
      // src sent is written before tgt sent
      bool src = true;
      if(nullToken.size() > 0) {
        // insert null token at the beginning of src sentence
        srcSents[lineNumber].push_back(Encode(nullToken));
      }
      for(unsigned i = 0; i < temp.size(); i++) {
        if(splits[i] == "|||") {
          // done with src sent. 
          src = false;
          // will now read tgt sent.
          continue;
        }
        if(src) {
          if(!reverse) {
            srcSents[lineNumber].push_back(temp[i]);
          } else {
            tgtSents[lineNumber].push_back(temp[i]);
          }
        } else {
          if(!reverse) {
            tgtSents[lineNumber].push_back(temp[i]);
          } else {
            srcSents[lineNumber].push_back(temp[i]);
          }
        }
      }
    }
  }
  
  // read each line in the text file, encodes each sentence into vector<int> and appends it into 'data'
  // assumptions: data is empty
  void Read(const std::string &textFilename, vector<vector<int64_t> > &data) {
    
    assert(data.size() == 0);
    
    // open data file
    std::ifstream textFile(textFilename.c_str(), std::ios::in);
    
    // for each line
    std::string line;
    int64_t lineNumber = -1;
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
    for(auto intToTokenIter = intToToken->begin(); intToTokenIter != intToToken->end(); intToTokenIter++) {
      bool inClosedVocab = closedVocab.find(intToTokenIter->first) != closedVocab.end();
      // c for closed, o for open
      vocabFile << intToTokenIter->first << " " << intToTokenIter->second << " " << (inClosedVocab? "c" : "o") << endl;
    }
    vocabFile.close();
  }

  const std::string Decode(int64_t wordId) const {
    if(intToToken->count(wordId) == 0) {
      return this->UNK;
    } else {
      return intToToken->find(wordId)->second.c_str();
    }
  }
  
  const std::string Decode(std::vector<int64_t> &wordIds) const {
    stringstream ss;
    for(auto wordIdIter = wordIds.begin(); wordIdIter != wordIds.end(); ++wordIdIter) {
      ss << Decode(*wordIdIter) << " ";
    }
    return ss.str();
  } 
  
  int64_t Count() const {
    return intToToken->size();
  }

  void* MapToSharedMemory(bool create, const string objectNickname) {
    cerr << "rank " << learningInfo.mpiWorld->rank() << ": entering VocabEncoder::MapToSharedMemory()" << endl;
    if(string(objectNickname) == string("VocabEncoder::tokenToInt")) {
      ShmemTokenToIntPairAllocator allocator(learningInfo.sharedMemorySegment->get_segment_manager()); 
      if(create) {
        cerr << "constructing VocabEncoder::tokenToInt...";
        auto temp = learningInfo.sharedMemorySegment->construct<ShmemTokenToIntMap> (objectNickname.c_str()) (std::less<char_string>(), allocator);
        cerr << "done." << endl;
        return temp;
      } else {
        cerr << "finding VocabEncoder::tokenToInt...";
        auto temp = learningInfo.sharedMemorySegment->find<ShmemTokenToIntMap> (objectNickname.c_str()).first;
        cerr << "done." << endl;
        return temp;
      }
    } else if (string(objectNickname) == string("VocabEncoder::intToToken")) {
      ShmemIntToTokenPairAllocator allocator(learningInfo.sharedMemorySegment->get_segment_manager()); 
      if(create) {
        cerr << "constructing VocabEncoder::intToToken...";
        auto temp = learningInfo.sharedMemorySegment->construct<ShmemIntToTokenMap> (objectNickname.c_str()) (std::less<int64_t>(), allocator);
        cerr << "done.";
        return temp;
      } else {
        cerr << "finding VocabEncoder::intToToken...";
        auto temp = learningInfo.sharedMemorySegment->find<ShmemIntToTokenMap> (objectNickname.c_str()).first;
        cerr << "done.";
        return temp;
      }
    } else {
      assert(false);
    }
    cerr << "rank " << learningInfo.mpiWorld->rank() << ": exiting VocabEncoder::MapToSharedMemory()" << endl;
    
  }
};

#endif

