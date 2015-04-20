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
#include "../wammar-utils/StringUtils.h"

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

typedef std::pair<const int64_t, int64_t> IntToIntPair;
typedef const boost::interprocess::allocator<IntToIntPair, boost::interprocess::managed_shared_memory::segment_manager> 
  ShmemIntToIntPairAllocator;
typedef boost::interprocess::map<int64_t, int64_t, std::less<int64_t>, ShmemIntToIntPairAllocator> 
  ShmemIntToIntMap;

typedef boost::interprocess::allocator<char_string, boost::interprocess::managed_shared_memory::segment_manager> 
  ShmemStringAllocator;
typedef vector<char_string, ShmemStringAllocator> 
  ShmemStringVector;

class VocabEncoder {
 public:

  // set with initialization list
  const LearningInfo &learningInfo;
  const int64_t firstId;
  const std::string UNK;
  
  // set for master and slaves
  void_allocator *alloc_inst;
  ShmemTokenToIntMap *tokenToInt;
  ShmemIntToTokenMap *intToToken;
  ShmemIntToIntMap *encodingToCount;
  
  // only set for master
  std::set<int64_t> closedVocab;
  char_string *UNK_char_string;

  bool countFrequencies;
  
 public:

  void Init() {
    
    alloc_inst = new void_allocator(learningInfo.sharedMemorySegment->get_segment_manager());
    
    // create/find managed shared memory objects
    if(learningInfo.mpiWorld->rank() == 0) {
      
      // create
      intToToken = (ShmemIntToTokenMap *) MapToSharedMemory(true, "VocabEncoder::intToToken");
      tokenToInt = (ShmemTokenToIntMap *) MapToSharedMemory(true, "VocabEncoder::tokenToInt");
      encodingToCount = (ShmemIntToIntMap *) MapToSharedMemory(true, "VocabEncoder::encodingToCount");
      UNK_char_string = (char_string *) MapToSharedMemory(true, "VocabEncoder::UNK_char_string");
      UNK_char_string->assign(UNK.c_str());
      
      // encode unk 
      tokenToInt->insert(TokenToIntPair(*UNK_char_string, firstId + intToToken->size()));
      intToToken->insert(IntToTokenPair(firstId + intToToken->size(), *UNK_char_string));
      
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
      encodingToCount = (ShmemIntToIntMap *) MapToSharedMemory(false, "VocabEncoder::encodingToCount");
      UNK_char_string = (char_string *) MapToSharedMemory(false, "VocabEncoder::UNK_char_string");
    }
    
  }
  
 VocabEncoder(const LearningInfo &learningInfo, unsigned firstId = 2): learningInfo(learningInfo), firstId(firstId), UNK("_unk_") {
    Init();
  }

 VocabEncoder(const std::string& textFilename, const LearningInfo &learningInfo, unsigned firstId = 2, unsigned minFreq = 1) : learningInfo(learningInfo), firstId(firstId), UNK("_unk_") {

    assert(minFreq >= 1);
    
    countFrequencies = true;
    Init();
    
    if(learningInfo.mpiWorld->rank() == 0) {
      
      cerr << learningInfo.mpiWorld->rank() << ": reading the vocabencoder init file " << textFilename <<  " now...";
      cerr << "minFreq = " << minFreq << endl;
      // create token-int correspondnence
      std::ifstream textFile(textFilename.c_str(), std::ios::in);
      std::string line;
      boost::unordered_map<string, int64_t> typeFrequency;
      
      // this pass, we only count the frequencies
      std::vector<string> splits;
      if (minFreq > 1) { 
	while(getline(textFile, line)) {
	  splits.clear();
	  if(line.size() == 0) { continue; }
	  StringUtils::SplitString(line, ' ', splits);
	  for(std::vector<string>::const_iterator tokenIter = splits.begin(); 
	      tokenIter != splits.end();
	      tokenIter++) {
	    if(typeFrequency.find(*tokenIter) == typeFrequency.end()) { typeFrequency[*tokenIter] = 0; }
	    typeFrequency[*tokenIter] += 1;
	  }
	}
      }
      textFile.close();
      
      // this pass, we actually encode strings
      textFile.open(textFilename.c_str(), std::ios::in);
      while(getline(textFile, line)) {
        if(line.size() == 0) {
          continue;
        }
        std::vector<string> splits;
        StringUtils::SplitString(line, ' ', splits);
        for (std::vector<string>::const_iterator tokenIter = splits.begin(); 
            tokenIter != splits.end();
            tokenIter++) {
          int temp = Encode(*tokenIter);
          // if this string is not frequent enough, modify its encoding to UNK
          if (minFreq > 1 && typeFrequency[*tokenIter] < minFreq) {
            char_string token_char_string(tokenIter->c_str(), *alloc_inst);
            tokenToInt->find(token_char_string)->second = UnkInt();
            cerr << " => " << tokenToInt->find(token_char_string)->second;
          }
        }
      }
      cerr << "done reading." << endl;
    }
    
    bool dummy;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, dummy, 0);
    countFrequencies = false;
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

  string UnkString() const {
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

  inline int64_t GetFrequencyCount(const int64_t encoding) {
    return encodingToCount->find(encoding)->second;
  }

  int64_t Encode(const string& token) {

    try {
      
      assert(alloc_inst != 0);
      char_string token_char_string(token.c_str(), *alloc_inst);

      if((*tokenToInt).count(token_char_string) == 0) {
        // slaves are not supposed to modify the shared objects
        assert(learningInfo.mpiWorld->rank() == 0);
        auto nextId = firstId + intToToken->size();
        tokenToInt->insert(TokenToIntPair(token_char_string, nextId));
        intToToken->insert(IntToTokenPair(nextId, token_char_string));
        if(countFrequencies) {
          encodingToCount->insert(IntToIntPair(nextId, 1));
        }
        return nextId;
      } else {
        auto encoding = tokenToInt->find(token_char_string)->second;
        if(learningInfo.mpiWorld->rank() == 0 && countFrequencies) {
          encodingToCount->find(encoding)->second++;
        }
        return encoding;
      }

    } catch(std::exception const&  ex) {

      cerr << "exception thrown inside Encode() with token " << token << " and process #" << learningInfo.mpiWorld->rank() << ". details: " << ex.what() << endl;
      assert(false);

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
  void ReadParallelCorpus(const std::string &textFilename, 
			  vector<vector<int64_t> > &srcSents, 
			  vector<vector<int64_t> > &tgtSents, 
			  const string &nullToken, bool reverse) {

    assert(srcSents.size() == 0 && tgtSents.size() == 0);

    Encode(nullToken); 
    
    // open data file
    std::ifstream textFile;
    if (learningInfo.mpiWorld->rank() == 0) {
      textFile.open(textFilename.c_str(), std::ios::in);
    }

    // for each line
    std::string line;
    int lineNumber = -1;
    bool end_of_file = false;
    while(true) {

      // read line
      if (learningInfo.mpiWorld->rank() == 0) {
	end_of_file = !getline(textFile, line);
      }
      boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, end_of_file, 0);

      if (end_of_file) {
	break;
      } else {
	boost::mpi::broadcast<std::string>(*learningInfo.mpiWorld, line, 0);
      }
      
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
        srcSents[lineNumber].push_back(ConstEncode(nullToken));
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
  
  void ReadConll(const std::string &conllFilename, vector< vector<ObservationDetails> > &data, unsigned minFreq = 1) {

    assert(minFreq >= 1);
    assert(data.size() == 0);    
    std::ifstream conllFile(conllFilename.c_str(), std::ios::in);
    std::string line;

    // every time a field is encoded, a frequency counter is incremented
    countFrequencies = true;

    unsigned sentIndex = 0;
    unsigned tokenIndex = 0;
    // make room for the first sentence
    data.resize(10000);
    boost::unordered_map<string, int64_t> typeFrequency;
    // this pass only computes the frequency of each wordtype
    while(getline(conllFile, line)) {
      std::vector<std::string> splits;
      StringUtils::SplitString(line, '\t', splits);
      if(splits.size() == 0) { continue; }
      for(unsigned i = 0; i < splits.size(); ++i) {
        if(typeFrequency.find(splits[i]) == typeFrequency.end()) {
          typeFrequency[splits[i]] = 0;
        }
        typeFrequency[splits[i]] += 1;        
      }
    }
    conllFile.close();
    conllFile.open(conllFilename.c_str(), std::ios::in);
    // this pass is the real encoding work
    while(getline(conllFile, line)) {
      std::vector<std::string> splits;
      StringUtils::SplitString(line, '\t', splits);
      // sanity check
      assert(tokenIndex == data[sentIndex].size());
        
      if(splits.size() == 0) {
        // update indexes
        sentIndex += 1;
        tokenIndex = 0;
        // make room for next sentence
        if(sentIndex + 1 > data.size()) {
          data.resize(data.size()+10000);
        }
      } else {
        // encode the splits
        vector<int64_t> encodedSplits;
        Encode(splits, encodedSplits);
        
        for(unsigned i = 0; i < splits.size(); ++i) {
          // if this string is not frequent enough, modify its encoding to UNK
          if(typeFrequency[splits[i]] < minFreq) {
            char_string token_char_string(splits[i].c_str(), *alloc_inst);
            tokenToInt->find(token_char_string)->second = UnkInt();
            encodedSplits[i] = UnkInt();
          }
        }
        
        // replace the integral fields with their actual value instead of their vocab id
        encodedSplits[ObservationDetailsHeader::ID] = (int64_t)stoi(splits[ObservationDetailsHeader::ID]);
        encodedSplits[ObservationDetailsHeader::HEAD] = (int64_t)stoi(splits[ObservationDetailsHeader::HEAD]);
        // update indexes
        tokenIndex += 1;
        // add a complex observation to the current sentence
        ObservationDetails tokenDetails(encodedSplits);
        data[sentIndex].push_back(tokenDetails);
      }
    }
    data.resize(sentIndex);
    assert( data[data.size()-1].size() > 0 );
    countFrequencies = false;
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
    if(string(objectNickname) == string("VocabEncoder::tokenToInt")) {
      if(create) {
        ShmemTokenToIntPairAllocator allocator(learningInfo.sharedMemorySegment->get_segment_manager()); 
        auto temp = learningInfo.sharedMemorySegment->construct<ShmemTokenToIntMap> (objectNickname.c_str()) (std::less<char_string>(), allocator);
        assert(temp);
        return temp;
      } else {
        auto temp = learningInfo.sharedMemorySegment->find<ShmemTokenToIntMap> (objectNickname.c_str()).first;
        assert(temp);
        return temp;
      }
    } else if (string(objectNickname) == string("VocabEncoder::encodingToCount")) {
      if(create) {
        ShmemIntToIntPairAllocator allocator(learningInfo.sharedMemorySegment->get_segment_manager()); 
        auto temp = learningInfo.sharedMemorySegment->construct<ShmemIntToIntMap> (objectNickname.c_str()) (std::less<int64_t>(), allocator);
        assert(temp);
        return temp;
      } else {
        auto temp = learningInfo.sharedMemorySegment->find<ShmemIntToIntMap> (objectNickname.c_str()).first;
        assert(temp);
        return temp;
      }
    } else if (string(objectNickname) == string("VocabEncoder::intToToken")) {
      if(create) {
        ShmemIntToTokenPairAllocator allocator(learningInfo.sharedMemorySegment->get_segment_manager()); 
        auto temp = learningInfo.sharedMemorySegment->construct<ShmemIntToTokenMap> 
          (objectNickname.c_str()) 
          (std::less<int64_t>(), allocator);
        assert(temp);
        return temp;
      } else {
        auto temp = learningInfo.sharedMemorySegment->find<ShmemIntToTokenMap> (objectNickname.c_str()).first;
        assert(temp);
        return temp;
      }
    } else if (string(objectNickname) == string("VocabEncoder::UNK_char_string")) {
      if(create) {
        char_allocator allocator(learningInfo.sharedMemorySegment->get_segment_manager());
        auto temp = learningInfo.sharedMemorySegment->construct<char_string> 
          (objectNickname.c_str()) 
          (allocator);
        assert(temp);
        return temp;
      } else {
        auto temp = learningInfo.sharedMemorySegment->find<char_string> (objectNickname.c_str()).first;
        assert(temp);
        return temp;
      }
    } else {
      assert(false);
    }

  }
};

#endif

