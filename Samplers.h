#ifndef _ALIAS_SAMPLER_H_
#define _ALIAS_SAMPLER_H_

#include <vector>
#include <limits>
#include <cstdlib>
#include <assert.h>

// continuous uniform sampler with range [0,maxValue]
struct UniformSampler {

  double maxValue;
  UniformSampler(double maxValue = 1.0) {
    this->maxValue = maxValue;
  }

  double Draw() const {
    // generate a random number between 0 and maxValue
    double randomScore = ((double) rand() / (RAND_MAX)) * maxValue;
    return randomScore;
  }
};

// dumb multinomial sampler with range {0,...,probs.size()-1}
// note: the probabilities need not be normalized, but none of them is allowed to be zero (TODO: allow zeros)
struct MultinomialSampler {
  std::vector<double> probs;
  double totalProb;
  UniformSampler uniformSampler;

  MultinomialSampler(std::vector<double>& probs) {
    this->probs = probs;
    totalProb = 0;
    for(int i = 0; i < probs.size(); i++) {
      assert(probs[i] > 0);
      totalProb += probs[i];
    }
  }
  
  virtual unsigned Draw() const {
    double shoot = uniformSampler.Draw() * totalProb;
    for(int i = 0; i < probs.size(); i++) {
      if(probs[i] >= shoot) {
	return i;
      } else {
	shoot -= probs[i];
      }
    }
    assert(false);
  }
};

// rich in the sense that users can specify values for the multinomial variable, instead of assuming the first 
// probability corresponds to value 0, next to value 1, ...etc.
struct RichMultinomialSampler : public MultinomialSampler {
  std::vector<unsigned> labels;
  
 RichMultinomialSampler(std::vector<double>& probs, std::vector<unsigned>& labels) 
   : MultinomialSampler(probs, copyParams) {
    assert(probs.size() == labels.size());
    this->labels = labels;
  }
  
  virtual unsigned Draw() const {
    unsigned labelIndex = MultinomialSampler::Draw();
    assert(labelIndex < labels.size());
    return labels[labelIndex];
  }
};

// R. A. Kronmal and A. V. Peterson, Jr. (1977) On the alias method for
// generating random variables from a discrete distribution. In The American
// Statistician, Vol. 33, No. 4. Pages 214--218.
//
// Intuition: a multinomial with N outcomes can be rewritten as a uniform
// mixture of N Bernoulli distributions. The ith Bernoulli returns i with
// probability F[i], otherwise it returns an "alias" value L[i]. The
// constructor computes the F's and L's given an arbitrary multionimial p in
// O(n) time and Draw returns samples in O(1) time.
struct AliasSampler {

  AliasSampler() {}
  explicit AliasSampler(const std::vector<double>& p) { Init(p); }

  void Init(const std::vector<double>& p) {
    const unsigned N = p.size();
    cutoffs_.resize(p.size());
    aliases_.clear();
    aliases_.resize(p.size(), std::numeric_limits<unsigned>::max());
    std::vector<unsigned> s,g;
    for (unsigned i = 0; i < N; ++i) {
      const double cutoff = cutoffs_[i] = N * p[i];
      if (cutoff >= 1.0) g.push_back(i); else s.push_back(i);
    }
    while(!s.empty() && !g.empty()) {
      const unsigned k = g.back();
      const unsigned j = s.back();
      aliases_[j] = k;
      cutoffs_[k] -= 1.0 - cutoffs_[j];
      s.pop_back();
      if (cutoffs_[k] < 1.0) {
        g.pop_back();
        s.push_back(k);
      }
    }
  }

  unsigned Draw() const {
    const unsigned n = uniformSampler.Draw() * cutoffs_.size();
    if (uniformSampler.Draw() > cutoffs_[n]) return aliases_[n]; else return n;
  }

  std::vector<double> cutoffs_;    // F
  std::vector<unsigned> aliases_;  // L
  UniformSampler uniformSampler;
};

#endif
