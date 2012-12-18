#ifndef _FST_UTILS_H_	  
#define _FST_UTILS_H_

#include <ctime>
#include <cstdlib>
#include <fst/fstlib.h>
#include <typeinfo>

#include "Samplers.h"

// pair typedef
typedef fst::ProductWeight<fst::LogWeight, fst::LogWeight> LogPairWeight;
typedef fst::ProductArc<fst::LogWeight, fst::LogWeight> LogPairArc;

// triple typedef
typedef fst::ProductWeight<LogPairWeight, fst::LogWeight> LogTripleWeight;
typedef fst::ProductArc<LogPairWeight, fst::LogWeight> LogTripleArc;

// quadruple typedef
typedef fst::ProductWeight<LogTripleWeight, fst::LogWeight> LogQuadWeight;
typedef fst::ProductArc<LogTripleWeight, fst::LogWeight> LogQuadArc;

class FstUtils {
 public:
  static const int LOG_ZERO = 30;
  static const int EPSILON = 0;
  static const float LOG_PROBS_MUST_BE_GREATER_THAN_ME; // set in FstUtils.cc

  inline static float nLog(double prob) {
    return -1.0 * log(prob);
  }

  inline static double nExp(float exponent) {
    return exp(-1.0 * exponent);
  }

  static void SampleFst(const fst::VectorFst<LogQuadArc>& fst, fst::VectorFst<LogQuadArc>& sampledFst, int sampleSize);

  static bool AreShadowFsts(const fst::VectorFst<LogQuadArc>& fst1, const fst::VectorFst<fst::LogArc>& fst2);

  static int FindFinalState(const fst::VectorFst<LogQuadArc>& fst);
  static int FindFinalState(const fst::VectorFst<fst::LogArc>& fst);

  static void MakeOneFinalState(fst::VectorFst<fst::LogArc>& fst);
  static void MakeOneFinalState(fst::VectorFst<LogQuadArc>& fst);
  
  static LogPairWeight EncodePair(float val1, float val2); 
  static LogTripleWeight EncodeTriple(float val1, float val2, float val3);
  static LogQuadWeight EncodeQuad(float val1, float val2, float val3, float val4);

  static LogPairWeight EncodePairInfinity();
  static LogTripleWeight EncodeTripleInfinity();
  static LogQuadWeight EncodeQuadInfinity();

  static void DecodePair(const LogPairWeight& w, float& v1, float& v2);
  static void DecodeTriple(const LogTripleWeight& w, float& v1, float& v2, float& v3);
  static void DecodeQuad(const LogQuadWeight& w, float& v1, float& v2, float& v3, float& v4);

  static string PrintAlignment(const fst::VectorFst< fst::StdArc > &bestAlignment);

  static string PrintWeight(const fst::TropicalWeightTpl<float>& w);
  static string PrintWeight(const fst::LogWeight& w);
  static string PrintWeight(const LogPairWeight& w);
  static string PrintWeight(const LogTripleWeight& w);
  static string PrintWeight(const LogQuadWeight& w);

  template<class ArcType>
    inline static string PrintFstSummary(const fst::VectorFst<ArcType>& fst) {
    std::stringstream ss;
    ss << "=======" << endl;
    ss << "states:" << endl;
    ss << "=======" << endl << endl;
    for(fst::StateIterator< fst::VectorFst<ArcType> > siter(fst); !siter.Done(); siter.Next()) {
    const int &stateId = siter.Value();
    string initial = fst.Start() == stateId? " START " : "";
    ss << "state:" << stateId << initial << " FinalScore=" <<  PrintWeight(fst.Final(stateId)) << endl;
    ss << "arcs:" << endl;
    for(fst::ArcIterator< fst::VectorFst<ArcType> > aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
      const ArcType &arc = aiter.Value();
      ss << arc.ilabel << ":" << arc.olabel << " " <<  stateId;
      ss << "-->" << arc.nextstate << " " << PrintWeight(arc.weight) << endl;
    } 
    ss << endl;
    }
    return ss.str(); 
  }

  template<class WeightType, class ArcType>
    inline static void ComputeTotalProb(const fst::VectorFst<ArcType>& prob, fst::VectorFst<ArcType>& totalProbs, WeightType& beta0) {

    // get the potentials from the initial state (i.e. total probabilty of getting from the initial state to each state)
    std::vector< WeightType > alphas;
    //cerr << "computing alphas" << endl;
    fst::ShortestDistance(prob, &alphas, false);
    
    // get the potentials to the final state (i.e. total probability of getting from each state to a final state)
    std::vector< WeightType > betas;
    //cerr << "computing betas" << endl;
    fst::ShortestDistance(prob, &betas, true);
    //cerr << "setting beta0 = betas[" << prob.Start() << "]" << endl;
    beta0 = betas[prob.Start()];
    
    // create states in the totalProbs FST, identical to those in prob
    assert(totalProbs.NumStates() == 0 && prob.NumStates() > 0);
    //cerr << "creating states of the shadow fst" << endl;
    while(totalProbs.NumStates() < prob.NumStates()) {
      int stateId = totalProbs.AddState();
      if(prob.Final(stateId) != WeightType::Zero()) {
	totalProbs.SetFinal(stateId, WeightType::One());
      }
    }
    totalProbs.SetStart(prob.Start());
      
    // compute the end-to-end probability for traversing each arc in prob
    //cerr << "creating arcs of the shadow fst" << endl;
    for(fst::StateIterator< fst::VectorFst<ArcType> > siter(prob); !siter.Done(); siter.Next()) {
      typename ArcType::StateId from = siter.Value();
      for(fst::ArcIterator< fst::VectorFst<ArcType> > aiter(prob, from); !aiter.Done(); aiter.Next()) {
	ArcType arc = aiter.Value();
	typename ArcType::StateId to = arc.nextstate;
	// posterior arc weight = weight(start=>arc) * weight(arc) * weight(arc=>final)
	WeightType arcTotalProb = fst::Divide(fst::Times(fst::Times(alphas[from], arc.weight), betas[to]), beta0);
	// TODO: assert this is a valid probability (it's tricky because we don't know which semiring is used.
	//	if(typeid(WeightType).name() == "N3fst12LogWeightTplIfEE") {
	//	  assert(arcTotalProb >= 0);
	//	}
	// add the arc
	totalProbs.AddArc(from, ArcType(arc.ilabel, arc.olabel, arcTotalProb, to));
      } 
    }

    // for debugging only
    //    cerr << "===============TotalProbFst===================" << endl;
    //    cerr << FstUtils::PrintFstSummary(totalProbs) << endl << endl;
  }
};

// an arc mapper that doesn't change anything in the FST layout, but replaces each LogTripleWeight 
// with a LogWeight equal to the third component in LogTripleWeight
struct LogTripleToLogMapper {
  fst::LogArc operator()(const LogTripleArc &arc) const {
    float v1, v2, v3;
    FstUtils::DecodeTriple(arc.weight, v1, v2, v3);
    return fst::LogArc(arc.ilabel, arc.olabel, v3, arc.nextstate);
  }
  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }
  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const { return props; }
};

// an arc mapper that doesn't change anything in the FST layout, but replaces each LogTripleWeight 
// with a LogWeight equal to the third component in LogTripleWeight
struct LogTripleToLogPositionMapper {
  fst::LogArc operator()(const LogTripleArc &arc) const {
    float v1, v2, v3;
    FstUtils::DecodeTriple(arc.weight, v1, v2, v3);
    return fst::LogArc(arc.ilabel, arc.olabel, v3, arc.nextstate);
  }
  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }
  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const { return props; }
};

// an arc mapper that doesn't change anything in the FST layout, but replaces each LogQuadWeight 
// with a LogWeight equal to the last component in LogQuadWeight
struct LogQuadToLogMapper {
  fst::LogArc operator()(const LogQuadArc &arc) const {
    float v1, v2, v3, v4;
    FstUtils::DecodeQuad(arc.weight, v1, v2, v3, v4);
    return fst::LogArc(arc.ilabel, arc.olabel, v4, arc.nextstate);
  }
  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }
  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const { return props; }
};

// an arc mapper that doesn't change anything in the FST layout, but replaces each LogQuadWeight 
// with a LogWeight equal to the last component in LogQuadWeight, and also changes the input/output
// labels on each arc from tgtToken/srcToken to tgtPos/srcPos
struct LogQuadToLogPositionMapper {
  fst::LogArc operator()(const LogQuadArc &arc) const {
    float tgtPos, srcPos, v3, logprob;
    FstUtils::DecodeQuad(arc.weight, tgtPos, srcPos, v3, logprob);
    if(arc.ilabel == FstUtils::EPSILON && arc.olabel == FstUtils::EPSILON) {
      return fst::LogArc(FstUtils::EPSILON, FstUtils::EPSILON, logprob, arc.nextstate);
    }
    //    cerr << " arc was " << arc.ilabel << ":" << arc.olabel << " (" << tgtPos << ", " << srcPos << ", " << v3 << ", " << logprob << ") =>" << arc.nextstate << endl;
    //    cerr << " arc became " << (int)tgtPos << ":" << (int)srcPos << " (" << logprob << ") =>" << arc.nextstate << endl;
    return fst::LogArc((int)tgtPos, (int)srcPos, logprob, arc.nextstate);
  }
  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }
  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const { return props; }
};

// an arc mapper that doesn't change anything in the FST layout, but replaces each LogWeight
// with a TropicalWeight (which has the path property)
struct LogToTropicalMapper {
  fst::StdArc operator()(const fst::LogArc &arc) const {
    return fst::StdArc(arc.ilabel, arc.olabel, arc.weight.Value(), arc.nextstate);
  }
  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }
  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const { return props; }
};

// an arc mapper that doesn't change anything in the FST layout, but replaces each TropicalWeight
// with a LogWeight (which can be used to run forward backward)
struct TropicalToLogMapper {
  fst::LogArc operator()(const fst::StdArc &arc) const {
    return fst::LogArc(arc.ilabel, arc.olabel, arc.weight.Value(), arc.nextstate);
  }
  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }
  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const { return props; }
};

#endif
