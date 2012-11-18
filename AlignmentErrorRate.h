#ifndef _ALIGNMENT_ERROR_RATE_H_
#define _ALIGNMENT_ERROR_RATE_H_

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <assert.h>
#include <vector>
#include <set>

// usage: 
// - Reset() or instantiate a new object
// - for each sentence in the test set, call CompareAlignment(sureAlignment, possibleAlignments, automaticAlignment)
// - call ComputeAER(), the return value is 
class AlignmentErrorRate {
 public:

  const static int NOT_SURE = -1;

  AlignmentErrorRate() {    
    Reset();
  }

  void Reset() {
    a_s = 0;
    a_p = 0;
    a = 0;
    s = 0;
  }

  // sureAlignment, possibleAlignments and automaticAlignments are each a vector of length = target s'entence length
  // sureAlignment[tgtPos] = -1 => unspecified
  // sureAlignment[tgtPos] = 0  => null alignment
  // sureAlignment[tgtPos] = 4  => tgtPos aligns to srcPos = 4 in this sentence
  // automaticAlignment[tgtPos] >= 0 (it cannot be unspecified)
  // possibleAlignment[tgtPos] = set of src positions. when the set is empty, sureAlignment[tgtPos] is assumed to be in the set.
  void CompareAlignment(std::vector<int> sureAlignment, std::vector< std::set< int > > possibleAlignments, std::vector<int> automaticAlignment) {
    assert(sureAlignment.size() == possibleAlignments.size());
    assert(possibleAlignments.size() == automaticAlignment.size());

    std::vector<int>::const_iterator automaticAlignmentIter = automaticAlignment.begin();
    std::vector<int>::const_iterator sureAlignmentIter = sureAlignment.begin();
    std::vector< std::set<int> >::const_iterator possibleAlignmentsIter = possibleAlignments.begin();
    for(; 
	automaticAlignmentIter != automaticAlignment.end();
	automaticAlignmentIter++, sureAlignmentIter++, possibleAlignmentsIter++) {
      assert(*automaticAlignmentIter >= 0);
      a++;
      if(*sureAlignmentIter != AlignmentErrorRate::NOT_SURE) {
	s++;
	if(*sureAlignmentIter == *automaticAlignmentIter) {
	  a_s++;
	}
      }
      for(std::set<int>::const_iterator possibleCurrentAlignmentsIter = possibleAlignmentsIter->begin();
	  possibleCurrentAlignmentsIter != possibleAlignmentsIter->end();
	  possibleCurrentAlignmentsIter++) {
	assert(*possibleCurrentAlignmentsIter >= 0);
	if(*possibleCurrentAlignmentsIter == *automaticAlignmentIter) {
	  a_p++;
	  break;
	}
      }
    }
  }

  float ComputeAER() {
    return 1.0 - (a_s + a_p) / (float) (a + s);
  }

 private:
  int a_s, a_p, a, s;
};

#endif
