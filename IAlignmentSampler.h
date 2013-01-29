#ifndef _I_ALIGNMENT_SAMPLER_H_
#define _I_ALIGNMENT_SAMPLER_H_

#include <vector>

class IAlignmentSampler {
  public:
    // sample an alignment and a translation, of a particular length given the source sentence.
    //also, return the proposal distribution's probability for this sample.
    virtual void SampleATGivenS(const std::vector<int> &srcTokens, 
				int tgtLength, 
				std::vector<int> &tgtTokens, 
				std::vector<int> &alignments, 
				double &logProb) = 0;

    // sample an alignment given a source sentence and a its translation.
    virtual void SampleAGivenST(const std::vector<int> &srcTokens,
				const std::vector<int> &tgtTokens,
				std::vector<int> &alignments,
				double &logProb) = 0;
};

#endif
