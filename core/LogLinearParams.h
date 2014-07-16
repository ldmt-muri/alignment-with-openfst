#ifndef _LOG_LINEAR_PARAMS_H_
#define _LOG_LINEAR_PARAMS_H_

#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <functional>
#include <utility>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered_map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
//#include "boost/archive/binary_oarchive.hpp"
//#include "boost/archive/binary_iarchive.hpp"

#include "../wammar-utils/unordered_map_serialization.hpp"

#include "../cdec-utils/fast_sparse_vector.h"

#include "LearningInfo.h"
#include "VocabEncoder.h"
#include "../wammar-utils/Samplers.h"
#include "../wammar-utils/tuple.h"

struct LogLinearParamsException : public std::exception
{
  std::string s;
 LogLinearParamsException(std::string ss) : s(ss) {}
  ~LogLinearParamsException() throw () {} // Updated
  const char* what() const throw() { return s.c_str(); }
};

struct FeatureId {
public:
  static VocabEncoder *vocabEncoder;
  FeatureTemplate type;
  union {
    int64_t wordBias;
    struct { int displacement; int64_t word; int label; } emission;
    struct { unsigned current, previous; } bigram;
    int alignmentJump;
    struct { int alignmentJump; int64_t wordBias; } biasedAlignmentJump;
    struct { int64_t srcWord, tgtWord; } wordPair;
    struct { int64_t word1, word2, word3; } wordTriple;
    int64_t precomputed;
    struct { unsigned alignerId; bool compatible; } otherAligner;
    struct { int position; int label; } boundaryLabel;
  };

  friend inline std::istream& operator>>(std::istream& is, FeatureId& obj)
  {
    string featureIdString;
    is >> featureIdString;
    vector<string> splits;
    StringUtils::SplitString(featureIdString, '|', splits);
    string &temp = splits[0];
    stringstream tempSS;
    obj.type = 
      temp == "BOUNDARY_LABELS"? FeatureTemplate::BOUNDARY_LABELS:
      temp == "EMISSION"? FeatureTemplate::EMISSION:
      temp == "LABEL_BIGRAM"? FeatureTemplate::LABEL_BIGRAM:
      temp == "SRC_BIGRAM"? FeatureTemplate::SRC_BIGRAM:
      temp == "ALIGNMENT_JUMP"? FeatureTemplate::ALIGNMENT_JUMP:
      temp == "LOG_ALIGNMENT_JUMP"? FeatureTemplate::LOG_ALIGNMENT_JUMP:
      temp == "HEAD_POS"? FeatureTemplate::HEAD_POS:
      temp == "ALIGNMENT_JUMP_IS_ZERO"? FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
      temp == "SRC0_TGT0"? FeatureTemplate::SRC0_TGT0:
      temp == "HC_TOKEN"? FeatureTemplate::HC_TOKEN:
      temp == "CH_TOKEN"? FeatureTemplate::CH_TOKEN:
      temp == "HC_POS"? FeatureTemplate::HC_POS:
      temp == "CH_POS"? FeatureTemplate::CH_POS:
      temp == "HEAD_CHILD_TOKEN_SET"? FeatureTemplate::HEAD_CHILD_TOKEN_SET:
      temp == "HEAD_CHILD_POS_SET"? FeatureTemplate::HEAD_CHILD_POS_SET:
      temp == "PRECOMPUTED"? FeatureTemplate::PRECOMPUTED:
      temp == "DIAGONAL_DEVIATION"? FeatureTemplate::DIAGONAL_DEVIATION:
      temp == "SRC_WORD_BIAS"? FeatureTemplate::SRC_WORD_BIAS:
      temp == "CHILD_POS"? FeatureTemplate::CHILD_POS:
      temp == "POS_PAIR_DISTANCE"? FeatureTemplate::POS_PAIR_DISTANCE:
      temp == "HCX_POS"? FeatureTemplate::HCX_POS:
      temp == "HXC_POS"? FeatureTemplate::HXC_POS:
      temp == "XHC_POS"? FeatureTemplate::XHC_POS:
      temp == "CHX_POS"? FeatureTemplate::CHX_POS:
      temp == "CXH_POS"? FeatureTemplate::CXH_POS:
      temp == "XCH_POS"? FeatureTemplate::XCH_POS:
      temp == "HxXC_POS"? FeatureTemplate::HxXC_POS:
      temp == "HXxC_POS"? FeatureTemplate::HXxC_POS:
      temp == "CxXH_POS"? FeatureTemplate::CxXH_POS:
      temp == "CXxH_POS"? FeatureTemplate::CXxH_POS:
      temp == "SYNC_START"? FeatureTemplate::SYNC_START:
      temp == "SYNC_END"? FeatureTemplate::SYNC_END:
      temp == "OTHER_ALIGNERS"? FeatureTemplate::OTHER_ALIGNERS:
      temp == "NULL_ALIGNMENT"? FeatureTemplate::NULL_ALIGNMENT:
      temp == "NULL_ALIGNMENT_LENGTH_RATIO"? FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO:
      FeatureTemplate::NONE;

    if(obj.type == FeatureTemplate::NONE) {
      cerr << "ERROR: malformatted feature id. ///" << temp << "/// is not a valid FeatureTemplate name" << endl;
    }
    
    switch(obj.type) {
    case FeatureTemplate::BOUNDARY_LABELS:
      tempSS.str(splits[1]);
      tempSS >> obj.boundaryLabel.position;
      tempSS.str(splits[2]); tempSS  >> obj.boundaryLabel.label;
      break;
    case FeatureTemplate::EMISSION:
      tempSS.str(splits[1]); tempSS >> obj.emission.displacement; 
      tempSS.str(splits[2]); tempSS >> obj.emission.label;
      obj.emission.word = FeatureId::vocabEncoder->Encode(splits[3]);
      break;
    case FeatureTemplate::LABEL_BIGRAM:
      tempSS.str(splits[1]); tempSS >> obj.bigram.previous;
      tempSS.str(splits[2]); tempSS >> obj.bigram.current;
      break;
    case FeatureTemplate::SRC_BIGRAM:
      obj.bigram.previous = FeatureId::vocabEncoder->Encode(splits[1]);
      obj.bigram.current = FeatureId::vocabEncoder->Encode(splits[2]);
      break;
    case FeatureTemplate::ALIGNMENT_JUMP:
      tempSS.str(splits[1]); tempSS>> obj.alignmentJump;
      break;
    case FeatureTemplate::LOG_ALIGNMENT_JUMP:
      tempSS.str(splits[1]); tempSS>> obj.biasedAlignmentJump.alignmentJump;
      obj.biasedAlignmentJump.wordBias = FeatureId::vocabEncoder->Encode(splits[2]);
      break;
    case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
      tempSS.str(splits[1]); tempSS>> obj.alignmentJump;
      break;
    case FeatureTemplate::SRC0_TGT0:
      obj.wordPair.srcWord = FeatureId::vocabEncoder->Encode(splits[1]);
      obj.wordPair.tgtWord = FeatureId::vocabEncoder->Encode(splits[2]);
      break;
    case FeatureTemplate::HC_TOKEN:
    case FeatureTemplate::HC_POS:
    case FeatureTemplate::CH_TOKEN:
    case FeatureTemplate::CH_POS:
    case FeatureTemplate::HEAD_CHILD_TOKEN_SET:
    case FeatureTemplate::HEAD_CHILD_POS_SET:
      obj.wordPair.srcWord = FeatureId::vocabEncoder->Encode(splits[1]);
      obj.wordPair.tgtWord = FeatureId::vocabEncoder->Encode(splits[2]);
      break;
    case FeatureTemplate::PRECOMPUTED:
      obj.precomputed = FeatureId::vocabEncoder->Encode(splits[1]);
      break;
    case FeatureTemplate::DIAGONAL_DEVIATION:
      tempSS.str(splits[1]); tempSS>> obj.wordBias;
      break;
    case FeatureTemplate::SRC_WORD_BIAS:
    case FeatureTemplate::HEAD_POS:
    case FeatureTemplate::CHILD_POS:
      obj.wordBias = FeatureId::vocabEncoder->Encode(splits[1]);
      break;
    case FeatureTemplate::HCX_POS:
    case FeatureTemplate::CHX_POS:
    case FeatureTemplate::XHC_POS:
    case FeatureTemplate::XCH_POS:
    case FeatureTemplate::HXC_POS:
    case FeatureTemplate::CXH_POS:
    case FeatureTemplate::CXxH_POS:
    case FeatureTemplate::HXxC_POS:
    case FeatureTemplate::HxXC_POS:
    case FeatureTemplate::CxXH_POS:
    case FeatureTemplate::POS_PAIR_DISTANCE:
      obj.wordTriple.word1 = FeatureId::vocabEncoder->Encode(splits[1]);
      obj.wordTriple.word2 = FeatureId::vocabEncoder->Encode(splits[2]);
      obj.wordTriple.word3 = FeatureId::vocabEncoder->Encode(splits[3]);
      break;
    case FeatureTemplate::OTHER_ALIGNERS:
      tempSS.str(splits[1]); tempSS >> obj.otherAligner.alignerId; // otherAligner.alignerId
      tempSS.str(splits[2]); tempSS >> obj.otherAligner.compatible; //otherAligner.compatible;
      break;
    case FeatureTemplate::SYNC_START:
    case FeatureTemplate::SYNC_END:
    case FeatureTemplate::NULL_ALIGNMENT:
    case FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO:
      break;
    default:
      assert(false);
    }
    return is;
  }
  
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & type;
    switch(type) {
    case FeatureTemplate::BOUNDARY_LABELS:
      ar & boundaryLabel.position;
      ar & boundaryLabel.label;
      break;
    case FeatureTemplate::DIAGONAL_DEVIATION:
    case FeatureTemplate::SRC_WORD_BIAS:
    case FeatureTemplate::HEAD_POS:
    case FeatureTemplate::CHILD_POS:
      ar & wordBias;
      break;
    case FeatureTemplate::ALIGNMENT_JUMP:
    case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
      ar & alignmentJump;
      break;
    case FeatureTemplate::LOG_ALIGNMENT_JUMP:
      ar & biasedAlignmentJump.alignmentJump;
      ar & biasedAlignmentJump.wordBias;
      break;
      case FeatureTemplate::SYNC_END:
    case FeatureTemplate::SYNC_START:
    case FeatureTemplate::NULL_ALIGNMENT:
    case FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO:
      break;
    case FeatureTemplate::EMISSION:
      ar & emission.displacement;
      ar & emission.label;
      ar & emission.word;
      break;
    case FeatureTemplate::LABEL_BIGRAM:
    case FeatureTemplate::SRC_BIGRAM:
      ar & bigram.previous;
      ar & bigram.current;
      break;
    case FeatureTemplate::PRECOMPUTED:
      ar & precomputed;
        break;
    case FeatureTemplate::SRC0_TGT0:
    case FeatureTemplate::HC_TOKEN:
    case FeatureTemplate::HC_POS:
    case FeatureTemplate::CH_TOKEN:
    case FeatureTemplate::CH_POS:
    case FeatureTemplate::HEAD_CHILD_TOKEN_SET:
    case FeatureTemplate::HEAD_CHILD_POS_SET:
      ar & wordPair.srcWord;
      ar & wordPair.tgtWord;
      break;
    case FeatureTemplate::HXC_POS:
    case FeatureTemplate::CXH_POS:
    case FeatureTemplate::XHC_POS:
    case FeatureTemplate::XCH_POS:
    case FeatureTemplate::HCX_POS:
    case FeatureTemplate::CHX_POS:
    case HXxC_POS:
    case CXxH_POS:
    case HxXC_POS:
    case CxXH_POS:
    case POS_PAIR_DISTANCE:
      ar & wordTriple.word1;
      ar & wordTriple.word2;
      ar & wordTriple.word3;
      break;
    case FeatureTemplate::OTHER_ALIGNERS:
      ar & otherAligner.alignerId;
      ar & otherAligner.compatible;
      break;
    default:
      assert(false);
    }
  }
  
  bool operator<(const FeatureId& rhs) const {
    if(type < rhs.type) return true;
    switch(type) {
    case FeatureTemplate::BOUNDARY_LABELS:
      return boundaryLabel.position < rhs.boundaryLabel.position || \
        (boundaryLabel.position == rhs.boundaryLabel.position && boundaryLabel.label < rhs.boundaryLabel.label);
      break;
      case FeatureTemplate::EMISSION:
        if(emission.displacement != rhs.emission.displacement) {
          return emission.displacement < rhs.emission.displacement;
        } else if(emission.label != rhs.emission.label) {
          return emission.label < rhs.emission.label;
        } else if(emission.word != rhs.emission.word) {
          return emission.word < rhs.emission.word;
        } else {
          return false;
        }
        break;
      case FeatureTemplate::LABEL_BIGRAM:
      case FeatureTemplate::SRC_BIGRAM:
        return bigram.current < rhs.bigram.current || \
          (bigram.current == rhs.bigram.current && bigram.previous < rhs.bigram.previous);
        break;
      case FeatureTemplate::ALIGNMENT_JUMP:
      case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
        return alignmentJump < rhs.alignmentJump;
        break;
      case FeatureTemplate::LOG_ALIGNMENT_JUMP:
        return biasedAlignmentJump.alignmentJump < rhs.biasedAlignmentJump.alignmentJump || \
          (biasedAlignmentJump.alignmentJump == rhs.biasedAlignmentJump.alignmentJump && biasedAlignmentJump.wordBias < rhs.biasedAlignmentJump.wordBias);
        break;
      case SRC0_TGT0:
      case FeatureTemplate::HC_TOKEN:
      case FeatureTemplate::HC_POS:        
      case FeatureTemplate::CH_TOKEN:
      case FeatureTemplate::CH_POS:        
      case FeatureTemplate::HEAD_CHILD_TOKEN_SET:
      case FeatureTemplate::HEAD_CHILD_POS_SET:        
        return wordPair.srcWord < rhs.wordPair.srcWord ||               \
          (wordPair.srcWord == rhs.wordPair.srcWord && wordPair.tgtWord < rhs.wordPair.tgtWord);
        break;
    case FeatureTemplate::HXC_POS:
    case FeatureTemplate::CXH_POS:
    case FeatureTemplate::XHC_POS:
    case FeatureTemplate::XCH_POS:
    case FeatureTemplate::HCX_POS:
    case FeatureTemplate::CHX_POS:
    case HXxC_POS:
    case CXxH_POS:
    case HxXC_POS:
    case CxXH_POS:
    case POS_PAIR_DISTANCE:
      return wordTriple.word1 < rhs.wordTriple.word1 ||
        (wordTriple.word1 == rhs.wordTriple.word1 && wordTriple.word2 < rhs.wordTriple.word2) ||
        (wordTriple.word1 == rhs.wordTriple.word1 && wordTriple.word2 == rhs.wordTriple.word2 && wordTriple.word3 < rhs.wordTriple.word3);
      break;
    case PRECOMPUTED:
        return precomputed < rhs.precomputed;
        break;
      case DIAGONAL_DEVIATION:
      case SRC_WORD_BIAS:
      case FeatureTemplate::HEAD_POS:
      case FeatureTemplate::CHILD_POS:
        return wordBias < rhs.wordBias; 
        break;
      case SYNC_START:
      case SYNC_END:
      case NULL_ALIGNMENT:
      case NULL_ALIGNMENT_LENGTH_RATIO:
        return false;
        break;
    case OTHER_ALIGNERS:
      return otherAligner.alignerId < rhs.otherAligner.alignerId || \
        (otherAligner.alignerId == rhs.otherAligner.alignerId && otherAligner.compatible < rhs.otherAligner.compatible);
      
      default:
        assert(false);
    }
  }
  

  bool operator!=(const FeatureId& rhs) const {
    if(type != rhs.type) return true;
    switch(type) {
    case FeatureTemplate::BOUNDARY_LABELS:
      return boundaryLabel.position != rhs.boundaryLabel.position || \
        boundaryLabel.label != rhs.boundaryLabel.label;
      break;
    case FeatureTemplate::EMISSION:
      return emission.displacement != rhs.emission.displacement ||  \
        emission.label != rhs.emission.label ||                     \
        emission.word != rhs.emission.word;
      break;
    case FeatureTemplate::LABEL_BIGRAM:
    case FeatureTemplate::SRC_BIGRAM:
      return bigram.current != rhs.bigram.current || bigram.previous != rhs.bigram.previous;
      break;
    case FeatureTemplate::ALIGNMENT_JUMP:
    case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
      return alignmentJump != rhs.alignmentJump;
      break;
    case FeatureTemplate::LOG_ALIGNMENT_JUMP:
      return biasedAlignmentJump.alignmentJump != rhs.biasedAlignmentJump.alignmentJump || \
        biasedAlignmentJump.wordBias != rhs.biasedAlignmentJump.wordBias;
      break;
    case SRC0_TGT0:
    case HC_TOKEN:
    case HC_POS:
    case CH_TOKEN:
    case CH_POS:
    case HEAD_CHILD_TOKEN_SET:
    case HEAD_CHILD_POS_SET:
      return wordPair.srcWord != rhs.wordPair.srcWord || wordPair.tgtWord != rhs.wordPair.tgtWord;
      break;
    case FeatureTemplate::HXC_POS:
    case FeatureTemplate::CXH_POS:
    case FeatureTemplate::XHC_POS:
    case FeatureTemplate::XCH_POS:
    case FeatureTemplate::HCX_POS:
    case FeatureTemplate::CHX_POS:
    case FeatureTemplate::HXxC_POS:
    case FeatureTemplate::CXxH_POS:
    case FeatureTemplate::HxXC_POS:
    case FeatureTemplate::CxXH_POS:
    case FeatureTemplate::POS_PAIR_DISTANCE:
      return wordTriple.word1 != rhs.wordTriple.word1 || 
        wordTriple.word2 != rhs.wordTriple.word2 || 
        wordTriple.word3 != rhs.wordTriple.word3;
      break;
    case PRECOMPUTED:
      return precomputed != rhs.precomputed;
      break;
    case DIAGONAL_DEVIATION:
    case SRC_WORD_BIAS:
    case FeatureTemplate::HEAD_POS:
    case FeatureTemplate::CHILD_POS:
      return wordBias != rhs.wordBias;
      break;
    case SYNC_START:
    case SYNC_END:
    case NULL_ALIGNMENT:
    case NULL_ALIGNMENT_LENGTH_RATIO:
      return false;
      break;
    case OTHER_ALIGNERS:
      return otherAligner.alignerId != rhs.otherAligner.alignerId || otherAligner.compatible != rhs.otherAligner.compatible;
    default:
      assert(false);
    }
  }
  
  bool operator==(const FeatureId& rhs) const {
    return !this->operator !=(rhs);
  }

  struct FeatureIdHash : public std::unary_function<FeatureId, size_t> {
    size_t operator()(const FeatureId& x) const {
      size_t seed = 0;
      boost::hash_combine(seed, x.type);
      switch(x.type) {
      case FeatureTemplate::BOUNDARY_LABELS:
        boost::hash_combine(seed, x.boundaryLabel.position);
        boost::hash_combine(seed, x.boundaryLabel.label);
        break;
      case FeatureTemplate::EMISSION:
          boost::hash_combine(seed, x.emission.displacement);
          boost::hash_combine(seed, x.emission.label);
          boost::hash_combine(seed, x.emission.word);
          break;
        case FeatureTemplate::LABEL_BIGRAM:
        case FeatureTemplate::SRC_BIGRAM:
          boost::hash_combine(seed, x.bigram.current);
          boost::hash_combine(seed, x.bigram.previous);
          break;
        case FeatureTemplate::ALIGNMENT_JUMP:
        case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
          boost::hash_combine(seed, x.alignmentJump);
          break;
        case FeatureTemplate::LOG_ALIGNMENT_JUMP:
          boost::hash_combine(seed, x.biasedAlignmentJump.alignmentJump);
          boost::hash_combine(seed, x.biasedAlignmentJump.wordBias);
          break;
        case SRC0_TGT0:
        case HC_TOKEN:
        case HC_POS:
        case CH_TOKEN:
        case CH_POS:
        case HEAD_CHILD_TOKEN_SET:
        case HEAD_CHILD_POS_SET:
          boost::hash_combine(seed, x.wordPair.srcWord);
          boost::hash_combine(seed, x.wordPair.tgtWord);
          break;
      case HXC_POS:
      case CXH_POS:
      case XHC_POS:
      case XCH_POS:
      case HCX_POS:
      case CHX_POS:
      case HXxC_POS:
      case CXxH_POS:
      case HxXC_POS:
      case CxXH_POS:
      case POS_PAIR_DISTANCE:
          boost::hash_combine(seed, x.wordTriple.word1);
          boost::hash_combine(seed, x.wordTriple.word2);
          boost::hash_combine(seed, x.wordTriple.word3);
          break;
        case PRECOMPUTED:
          boost::hash_combine(seed, x.precomputed);
          break;
        case DIAGONAL_DEVIATION:
        case SRC_WORD_BIAS:
        case HEAD_POS:
        case CHILD_POS:
          boost::hash_combine(seed, x.wordBias);
          break;
        case SYNC_START:
        case SYNC_END:
        case NULL_ALIGNMENT:
	case NULL_ALIGNMENT_LENGTH_RATIO:
	  break;
        case OTHER_ALIGNERS:
	  boost::hash_combine(seed, x.otherAligner.alignerId);
	  boost::hash_combine(seed, x.otherAligner.compatible);
	  break;
        default:
          assert(false);
      }
      return seed;
    }
  };

  struct FeatureIdEqual : public std::unary_function<FeatureId, bool> {
    bool operator()(const FeatureId& left, const FeatureId& right) const {
      if(left.type != right.type) return false;
      switch(left.type) {
      case FeatureTemplate::BOUNDARY_LABELS:
        return left.boundaryLabel.position == right.boundaryLabel.position && \
          left.boundaryLabel.label == right.boundaryLabel.label;
        break;
      case FeatureTemplate::EMISSION:
        return left.emission.displacement == right.emission.displacement && \
          left.emission.label == right.emission.label &&                \
          left.emission.word == right.emission.word;
        break;
      case FeatureTemplate::LABEL_BIGRAM:
      case FeatureTemplate::SRC_BIGRAM:
        return left.bigram.current == right.bigram.current && left.bigram.previous == right.bigram.previous;
          break;
        case FeatureTemplate::ALIGNMENT_JUMP:
        case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
          return left.alignmentJump == right.alignmentJump;
          break;
        case FeatureTemplate::LOG_ALIGNMENT_JUMP:
          return left.biasedAlignmentJump.alignmentJump == right.biasedAlignmentJump.alignmentJump && \
            left.biasedAlignmentJump.wordBias == right.biasedAlignmentJump.wordBias;
          break;
        case FeatureTemplate::SRC0_TGT0:
        case FeatureTemplate::HC_TOKEN:
        case FeatureTemplate::HC_POS:
        case FeatureTemplate::CH_TOKEN:
        case FeatureTemplate::CH_POS:
        case FeatureTemplate::HEAD_CHILD_TOKEN_SET:
        case FeatureTemplate::HEAD_CHILD_POS_SET:
          return left.wordPair.srcWord == right.wordPair.srcWord && left.wordPair.tgtWord == right.wordPair.tgtWord;
          break;
      case HXC_POS:
      case CXH_POS:
      case HCX_POS:
      case CHX_POS:
      case XHC_POS:
      case XCH_POS:
      case HXxC_POS:
      case CXxH_POS:
      case HxXC_POS:
      case CxXH_POS:
      case POS_PAIR_DISTANCE:
        return left.wordTriple.word1 == right.wordTriple.word1 &&
          left.wordTriple.word2 == right.wordTriple.word2 &&
          left.wordTriple.word3 == right.wordTriple.word3;
        break;
        case FeatureTemplate::PRECOMPUTED:
          return left.precomputed == right.precomputed;
          break;
        case FeatureTemplate::DIAGONAL_DEVIATION:
        case FeatureTemplate::SRC_WORD_BIAS:
        case FeatureTemplate::HEAD_POS:
        case FeatureTemplate::CHILD_POS:
          return left.wordBias == right.wordBias;
          break;
        case FeatureTemplate::SYNC_START:
        case FeatureTemplate::SYNC_END:
        case FeatureTemplate::NULL_ALIGNMENT:
        case FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO:
          return true;
          break;
        case FeatureTemplate::OTHER_ALIGNERS:
	return left.otherAligner.alignerId == right.otherAligner.alignerId && \
	  left.otherAligner.compatible == right.otherAligner.compatible;
	  break;
        default:
          assert(false);
      }
    }
  };
  
};

// a few typedefs
using unordered_map_featureId_double = boost::unordered_map<FeatureId, double, FeatureId::FeatureIdHash, FeatureId::FeatureIdEqual>;
using unordered_map_featureId_int = boost::unordered_map<FeatureId, int, FeatureId::FeatureIdHash, FeatureId::FeatureIdEqual>;

// Alias an STL compatible allocator of ints that allocates ints from the managed
// shared memory segment.  This allocator will allow to place containers
// in managed shared memory segments
typedef boost::interprocess::allocator<double, boost::interprocess::managed_shared_memory::segment_manager> ShmemDoubleAllocator;
typedef boost::interprocess::allocator<FeatureId, boost::interprocess::managed_shared_memory::segment_manager> ShmemFeatureIdAllocator;
typedef FeatureId InnerKeyType;
typedef double InnerMappedType;
typedef std::pair<const InnerKeyType, InnerMappedType> InnerValueType;
typedef boost::interprocess::allocator<InnerValueType, boost::interprocess::managed_shared_memory::segment_manager> ShmemInnerValueAllocator;
typedef std::tuple<int, int> OuterKeyType;
typedef boost::interprocess::map<InnerKeyType, InnerMappedType, std::less<InnerKeyType>, ShmemInnerValueAllocator> OuterMappedType;
typedef std::pair<const OuterKeyType, OuterMappedType*> OuterValueType;
typedef boost::interprocess::allocator<OuterValueType, boost::interprocess::managed_shared_memory::segment_manager> ShmemOuterValueAllocator;
typedef std::map<OuterKeyType, OuterMappedType*, std::less<OuterKeyType>, ShmemOuterValueAllocator> ShmemNestedMap;

// Alias a vector that uses the previous STL-like allocator
typedef vector<double, ShmemDoubleAllocator> ShmemVectorOfDouble;
typedef vector<FeatureId, ShmemFeatureIdAllocator> ShmemVectorOfFeatureId;

std::ostream& operator<<(std::ostream& os, const FeatureId& obj);

class LogLinearParams {

  // inline and template member functions
#include "LogLinearParams-inl.h"

 public:

  // for the latent CRF model
  LogLinearParams(VocabEncoder &types, double gaussianStdDev = 1);
  
  OuterMappedType* MapWordPairFeaturesToSharedMemory(bool create, const std::pair<int64_t, int64_t> &wordPair);
  OuterMappedType* MapWordPairFeaturesToSharedMemory(bool create, const string& objectNickname);
  
  void* MapToSharedMemory(bool create, string name);

  template<class Archive>
    void save(Archive & os, const unsigned int version) const
    {
      assert(IsSealed());
      assert(paramIdsPtr->size() == paramWeightsPtr->size());
      int count = paramIdsPtr->size();
      os << count;
      for(unsigned i = 0; i < paramIdsPtr->size(); i++) {
        os << (*paramIdsPtr)[i];
        os << (*paramWeightsPtr)[i];
      }
    }
  
  template<class Archive>
    void load(Archive & is, const unsigned int version)
    {
      int count;
      is >> count;
      for(int i = 0; i < count; ++i) {
        FeatureId featureId;
        is >> featureId;
        paramIdsTemp.push_back(featureId);
        double weight;
        is >> weight;
        paramWeightsTemp.push_back(weight);
      }
    }

  BOOST_SERIALIZATION_SPLIT_MEMBER()  
    
  // this method seals the set of parameters being used, not their weights
  void Seal();
  void Unseal();
  bool IsSealed() const;

  void LoadPrecomputedFeaturesWith2Inputs(const std::string &wordPairFeaturesFilename);

  void AddToPrecomputedFeaturesWith2Inputs(int input1, int input2, FeatureId &featureId, double featureValue);

  double Hash();

  // set learning info
  void SetLearningInfo(LearningInfo &learningInfo);

  // load the word alignments when available
  void LoadOtherAlignersOutput();

  // given the description of one transition on the alignment FST, find the features that would fire along with their values
  void FireFeatures(int srcToken, int prevSrcToken, int tgtToken, 
		    int srcPos, int prevSrcPos, int tgtPos, 
		    int srcSentLength, int tgtSentLength, 
		    unordered_map_featureId_double& activeFeatures);

  // for pos tagging
  void FireFeatures(int yI, int yIM1, int sentId, const vector<int64_t> &x, unsigned i, 
		    FastSparseVector<double> &activeFeatures);
  
  // for word alignment
  void FireFeatures(int yI, int yIM1, const vector<int64_t> &x_t, const vector<int64_t> &x_s, unsigned i, 
		    int START_OF_SENTENCE_Y_VALUE, int NULL_POS,
		    FastSparseVector<double> &activeFeatures);

  // for dependency parsing
  void FireFeatures(const ObservationDetails &headDetails, const ObservationDetails &childDetails,
                    const std::vector<ObservationDetails> &sentDetails,
                    FastSparseVector<double> &activeFeatures);

  int AddParams(const std::vector< FeatureId > &paramIds);

  // if the paramId does not exist, add it with weight drawn from gaussian. otherwise, do nothing. 
  bool AddParam(const FeatureId &paramId);
  
  // if the paramId does not exist, add it. otherwise, do nothing. 
  bool AddParam(const FeatureId &paramId, double paramWeight);

  // side effect: adds zero weights for parameter IDs present in values but not present in paramIndexes and paramWeights
  double DotProduct(const unordered_map_featureId_double& values);

  double DotProduct(const std::vector<double>& values);

  double DotProduct(const std::vector<double>& values, const ShmemVectorOfDouble& weights);

  double DotProduct(const FastSparseVector<double> &values);

  double DotProduct(const FastSparseVector<double> &values, const ShmemVectorOfDouble& weights);
 
  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const unordered_map_featureId_double &gradient, const OptMethod &optMethod);
  
  // override the member weights vector with this array
  void UpdateParams(const double* array, const int arrayLength);
  
  // converts a map into an array. 
  // when constrainedFeaturesCount is non-zero, length(valuesArray)  should be = valuesMap.size() - constrainedFeaturesCount, 
  // we pretend as if the constrained features don't exist by subtracting the internal index - constrainedFeaturesCount  
  void ConvertFeatureMapToFeatureArray(unordered_map_featureId_double &valuesMap, double* valuesArray, unsigned constrainedFeaturesCount = 0);

  // 1/2 * sum of the squares 
  double ComputeL2Norm();
  
  // applies the cumulative l1 penalty on feature weights, also updates the appliedL1Penalty values 
  void ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere, 
   				LogLinearParams& appliedL1Penalty, 
   				const double correctL1Penalty); 

  void PrintFirstNParams(unsigned n); 
  
  void PrintParams(); 
  
  static void PrintParams(unordered_map_featureId_double &tempParams); 
  
  void PrintFeatureValues(FastSparseVector<double> &feats);

  // writes the features to a text file formatted one feature per line.  
  void PersistParams(const std::string& outputFilename, bool humanFriendly=true); 

  // loads the parameters
  void LoadParams(const std::string &inputFilename);

  // checks whether the "otherParams" have the same parameters and values as this object 
  // disclaimer: pretty expensive, and also requires that the parameters have the same order in the underlying vectors 
  bool IsIdentical(const LogLinearParams &otherParams);

  bool LogLinearParamsIsIdentical(const LogLinearParams &otherParams);

 public:
  // the actual parameters 
  unordered_map_featureId_int paramIndexes;
  ShmemVectorOfDouble *paramWeightsPtr; 
  std::vector< double > paramWeightsTemp; 
  ShmemVectorOfFeatureId *paramIdsPtr; 
  std::vector< FeatureId > paramIdsTemp; 
  
  // maps a word id into a string
  VocabEncoder &types;

  LearningInfo *learningInfo;

  const GaussianSampler *gaussianSampler;

  const set< int > *englishClosedClassTypes;

  // for each other aligner, for each sentence pair, for each target position, 
  // determines the corresponding src position 
  std::vector< std::vector< std::vector< std::set<int>* >* >* > otherAlignersOutput;

  boost::unordered_map< std::pair<int64_t, int64_t>, OuterMappedType*> cacheWordPairFeatures;
 
  boost::unordered_map< PosFactorId, FastSparseVector<double>, PosFactorId::PosFactorHash, PosFactorId::PosFactorEqual > posFactorIdToFeatures;
 
  unordered_map_featureId_double featureGaussianMeans;

  bool logging;

 private:
  bool sealed;
  
};

#endif
