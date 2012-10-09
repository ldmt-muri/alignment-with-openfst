#ifndef _FST_UTILS_H_
#define _FST_UTILS_H_

#include <fst/fstlib.h>

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

  inline static float nLog(float prob) {
    return -1.0 * log(prob);
  }
  inline static float nExp(float exponent) {
    return exp(-1.0 * exponent);
  }

  static void PrintFstSummary(fst::VectorFst<fst::LogArc>& fst);

  static LogPairWeight EncodePairInfinity();
  static LogPairWeight EncodePair(float val1, float val2);
  static void DecodePair(const LogPairWeight& w, float& v1, float& v2);
  static string PrintPair(const LogPairWeight& w);
  static void PrintFstSummary(fst::VectorFst<LogPairArc>& fst);
  
  static LogTripleWeight EncodeTripleInfinity();
  static LogTripleWeight EncodeTriple(float val1, float val2, float val3);
  static void DecodeTriple(const LogTripleWeight& w, float& v1, float& v2, float& v3);
  static string PrintTriple(const LogTripleWeight& w);
  static void PrintFstSummary(fst::VectorFst<LogTripleArc>& fst);

  static LogQuadWeight EncodeQuadInfinity();
  static LogQuadWeight EncodeQuad(float val1, float val2, float val3, float val4);
  static void DecodeQuad(const LogQuadWeight& w, float& v1, float& v2, float& v3, float& v4);
  static string PrintQuad(const LogQuadWeight& w);
  static void PrintFstSummary(fst::VectorFst<LogQuadArc>& fst);

  template<class WeightType, class ArcType>
    inline static void ComputeTotalProb(const fst::VectorFst<ArcType>& prob, fst::VectorFst<ArcType>& totalProbs, WeightType& beta0) {

    // for debugging
    //    cerr << "before:" << endl;
    //    cerr << "beta0=" << beta0.Value() << endl << endl;
    //    string dummy;

    // get the potentials from the initial state (i.e. total probabilty of getting from the initial state to each state)
    std::vector< WeightType > alphas;
    fst::ShortestDistance(prob, &alphas, false);
    
    // get the potentials to the final state (i.e. total probability of getting from each state to a final state)
    std::vector< WeightType > betas;
    fst::ShortestDistance(prob, &betas, true);
    beta0 = betas[prob.Start()];
    
    // create states in the totalProbs FST, identical to those in prob
    assert(totalProbs.NumStates() == 0 && prob.NumStates() > 0);
    while(totalProbs.NumStates() < prob.NumStates()) {
      int stateId = totalProbs.AddState();
      totalProbs.SetFinal(stateId, prob.Final(stateId));
    }
    totalProbs.SetStart(prob.Start());
      
    // compute the end-to-end probability for traversing each arc in prob
    for(fst::StateIterator< fst::VectorFst<ArcType> > siter(prob); !siter.Done(); siter.Next()) {
      typename ArcType::StateId from = siter.Value();
      for(fst::ArcIterator< fst::VectorFst<ArcType> > aiter(prob, from); !aiter.Done(); aiter.Next()) {
	ArcType arc = aiter.Value();
	typename ArcType::StateId to = arc.nextstate;
	WeightType arcTotalProb = fst::Times(fst::Times(alphas[from], arc.weight), betas[to]);
	totalProbs.AddArc(from, ArcType(arc.ilabel, arc.olabel, arcTotalProb, to));
      } 
    }

    // for debugging
    //    cerr << "====================posterior prob of arc=================" << endl;
    //    PrintFstSummary(totalProbs);
    //    string dummy;
    //    cerr << "after:" << endl;
    //    cerr << "beta0=" << beta0.Value() << endl;
    //    cin >> dummy;
  }
  
};

/*
namespace MapFinalAction {
  // This determines how final weights are mapped.  
  enum MapFinalAction { 
    // A final weight is mapped into a final weight. An error
    // is raised if this is not possible.  
    MAP_NO_SUPERFINAL,
    
    // A final weight is mapped to an arc to the superfinal state
    // when the result cannot be represented as a final weight.
    // The superfinal state will be added only if it is needed.  
    MAP_ALLOW_SUPERFINAL,
    
    // A final weight is mapped to an arc to the superfinal state
    // unless the result can be represented as a final weight of weight
    // Zero(). The superfinal state is always added (if the input is
    // not the empty Fst).  
    MAP_REQUIRE_SUPERFINAL
  };
}

namespace MapSymbolsAction {
  // This determines how symbol tables are mapped.  
  enum MapSymbolsAction { 
    // Symbols should be cleared in the result by the map.  
    MAP_CLEAR_SYMBOLS,
    
    // Symbols should be copied from the input FST by the map. 
    MAP_COPY_SYMBOLS,
    
    // Symbols should not be modified in the result by the map itself.
    // (They may set by the mapper). 
    MAP_NOOP_SYMBOLS
  };
}
*/

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
