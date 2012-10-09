#include "FstUtils.h"

using namespace fst;

LogPairWeight FstUtils::EncodePairInfinity() {
  return EncodePair(numeric_limits<float>::infinity(), numeric_limits<float>::infinity());
}

LogTripleWeight FstUtils::EncodeTripleInfinity() {
  return EncodeTriple(numeric_limits<float>::infinity(), numeric_limits<float>::infinity(), numeric_limits<float>::infinity());
}

LogQuadWeight FstUtils::EncodeQuadInfinity() {
  return EncodeQuad(numeric_limits<float>::infinity(), 
		    numeric_limits<float>::infinity(), 
		    numeric_limits<float>::infinity(), 
		    numeric_limits<float>::infinity());
}

LogPairWeight FstUtils::EncodePair(float val1, float val2) {
  LogWeight v1, v2;
  v1 = val1;
  v2 = val2;
  return ProductWeight<LogWeight, LogWeight>(v1, v2);
}

LogTripleWeight FstUtils::EncodeTriple(float val1, float val2, float val3) {
  LogWeight v3;
  v3 = val3;
  return LogTripleWeight(EncodePair(val1, val2), v3);
}

LogQuadWeight FstUtils::EncodeQuad(float val1, float val2, float val3, float val4) {
  LogWeight v4;
  v4 = val4;
  return LogQuadWeight(EncodeTriple(val1, val2, val3), v4);
}

string FstUtils::PrintPair(const ProductWeight<LogWeight, LogWeight>& w) {
  stringstream ss;
  ss << "(" << w.Value1().Value() << "," << w.Value2().Value() << ")";
  return ss.str();
}

string FstUtils::PrintTriple(const LogTripleWeight& w) {
  stringstream ss;
  ss << "(" << w.Value1().Value1().Value() 
     << "," << w.Value1().Value2().Value() 
     << "," << w.Value2().Value() 
     << ")";
  return ss.str();
}

string FstUtils::PrintQuad(const LogQuadWeight& w) {
  stringstream ss;
  ss << "(" << w.Value1().Value1().Value1().Value() 
     << "," << w.Value1().Value1().Value2().Value()
     << "," << w.Value1().Value2().Value()
     << "," << w.Value2().Value()
     << ")";
  return ss.str();
}

void FstUtils::DecodePair(const LogPairWeight& w, float& v1, float& v2) {
  v1 = w.Value1().Value();
  v2 = w.Value2().Value();
}

void FstUtils::DecodeTriple(const LogTripleWeight& w, float& v1, float& v2, float& v3) {
  DecodePair(w.Value1(), v1, v2);
  v3 = w.Value2().Value();
}

void FstUtils::DecodeQuad(const LogQuadWeight& w, float& v1, float& v2, float& v3, float& v4) {
  DecodeTriple(w.Value1(), v1, v2, v3);
  v4 = w.Value2().Value();
}

void FstUtils::PrintFstSummary(VectorFst<LogArc>& fst) {
  cout << "states:" << endl;
  for(StateIterator< VectorFst<LogArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogArc::StateId &stateId = siter.Value();
    string final = fst.Final(stateId) != 0? " FINAL": "";
    string initial = fst.Start() == stateId? " START" : "";
    cout << "state:" << stateId << initial << final << endl;
    cout << "arcs:" << endl;
    for(ArcIterator< VectorFst<LogArc> > aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
      const LogArc &arc = aiter.Value();
      cout << arc.ilabel << ":" << arc.olabel << " " <<  stateId << "-->" << arc.nextstate << " " << arc.weight << endl;
    } 
    cout << endl;
  }
}

void FstUtils::PrintFstSummary(VectorFst<LogPairArc>& fst) {
  cerr << "=======" << endl;
  cerr << "states:" << endl;
  cerr << "=======" << endl << endl;
  for(StateIterator< VectorFst<LogPairArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogPairArc::StateId &stateId = siter.Value();
    string initial = fst.Start() == stateId? " START " : "";
    cerr << "state:" << stateId << initial << " FinalScore=" <<  PrintPair(fst.Final(stateId)) << endl;
    cerr << "arcs:" << endl;
    for(ArcIterator< VectorFst<LogPairArc> > aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
      const LogPairArc &arc = aiter.Value();
      cerr << arc.ilabel << ":" << arc.olabel << " " <<  stateId;
      cerr << "-->" << arc.nextstate << " " << PrintPair(arc.weight) << endl;
    } 
    cerr << endl;
  }
}

void FstUtils::PrintFstSummary(VectorFst<LogTripleArc>& fst) {
  cerr << "=======" << endl;
  cerr << "states:" << endl;
  cerr << "=======" << endl << endl;
  for(StateIterator< VectorFst<LogTripleArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogTripleArc::StateId &stateId = siter.Value();
    string initial = fst.Start() == stateId? " START " : "";
    cerr << "state:" << stateId << initial << " FinalScore=" <<  PrintTriple(fst.Final(stateId)) << endl;
    cerr << "arcs:" << endl;
    for(ArcIterator< VectorFst<LogTripleArc> > aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
      const LogTripleArc &arc = aiter.Value();
      cerr << arc.ilabel << ":" << arc.olabel << " " <<  stateId;
      cerr << "-->" << arc.nextstate << " " << PrintTriple(arc.weight) << endl;
    } 
    cerr << endl;
  }
}

void FstUtils::PrintFstSummary(VectorFst<LogQuadArc>& fst) {
  cerr << "=======" << endl;
  cerr << "states:" << endl;
  cerr << "=======" << endl << endl;
  for(StateIterator< VectorFst<LogQuadArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogQuadArc::StateId &stateId = siter.Value();
    string initial = fst.Start() == stateId? " START " : "";
    cerr << "state:" << stateId << initial << " FinalScore=" <<  PrintQuad(fst.Final(stateId)) << endl;
    cerr << "arcs:" << endl;
    for(ArcIterator< VectorFst<LogQuadArc> > aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
      const LogQuadArc &arc = aiter.Value();
      cerr << arc.ilabel << ":" << arc.olabel << " " <<  stateId;
      cerr << "-->" << arc.nextstate << " " << PrintQuad(arc.weight) << endl;
    } 
    cerr << endl;
  }
}
