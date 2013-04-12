#ifndef _LATENT_CRF_MODEL_INL_H_
#define _LATENT_CRF_MODEL_INL_H_

// normalize soft counts with identical content to sum to one
template <typename ContextType>
void NormalizeThetaMle(MultinomialParams::ConditionalMultinomialParam<ContextType> &mle, 
		       std::map<ContextType, double> &mleMarginals) {
  // fix theta mle estimates
  for(typename std::map<ContextType, MultinomialParams::MultinomialParam >::const_iterator yIter = mle.params.begin(); yIter != mle.params.end(); yIter++) {
    ContextType y_ = yIter->first;
    double unnormalizedMarginalProbz_giveny_ = 0.0;
    // verify that \sum_z* mle[y*][z*] = mleMarginals[y*]
    for(MultinomialParams::MultinomialParam::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double unnormalizedProbz_giveny_ = zIter->second;
      unnormalizedMarginalProbz_giveny_ += unnormalizedProbz_giveny_;
    }
    if(abs((mleMarginals[y_] - unnormalizedMarginalProbz_giveny_) / mleMarginals[y_]) > 0.01) {
      cerr << "ERROR: abs( (mleMarginals[y_] - unnormalizedMarginalProbz_giveny_) / mleMarginals[y_] ) = ";
      cerr << abs((mleMarginals[y_] - unnormalizedMarginalProbz_giveny_) / mleMarginals[y_]) << endl; 
      cerr << "mleMarginals[y_] = " << mleMarginals[y_] << " unnormalizedMarginalProbz_giveny_ = " << unnormalizedMarginalProbz_giveny_ << endl;
      cerr << " --error ignored, but try to figure out what's wrong!" << endl;
    }
    // normalize the mle estimates to sum to one for each context
    for(MultinomialParams::MultinomialParam::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      assert(!std::isnan(mleMarginals[y_]) && !std::isinf(mleMarginals[y_]));
      assert(mleMarginals[y_] > zIter->second);
      double normalizedProbz_giveny_ = zIter->second / mleMarginals[y_];
      assert(!std::isnan(mle[y_][z_]) && !std::isinf(mle[y_][z_]));
      if(std::isnan(normalizedProbz_giveny_) || std::isinf(normalizedProbz_giveny_)) {
	cerr << "normalizedProbz_giveny_ = " << normalizedProbz_giveny_ << " = zIter->second / mleMarginals[y_] = " << zIter->second << " / " << mleMarginals[y_] << endl;
      }
      assert(!std::isnan(normalizedProbz_giveny_) && !std::isnan(normalizedProbz_giveny_));
      mle[y_][z_] = normalizedProbz_giveny_;
	
      // take the nlog
      mle[y_][z_] = MultinomialParams::nLog(mle[y_][z_]);
      assert(!std::isnan(mle[y_][z_]) && !std::isinf(mle[y_][z_]));

    }
  }
}

template <typename ContextType>
// returns -log p(z|x)
double UpdateThetaMleForSent(const unsigned sentId, 
			   MultinomialParams::ConditionalMultinomialParam<ContextType> &mle, 
			   std::map<ContextType, double> &mleMarginals) {
  if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
    std::cerr << "sentId = " << sentId << endl;
  }
  assert(sentId < examplesCount);
  // build the FSTs
  fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
  fst::VectorFst<FstUtils::LogArc> lambdaFst;
  std::vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
  BuildThetaLambdaFst(sentId, GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
  BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas);
  // compute the B matrix for this sentence
  std::map< ContextType, std::map< int, LogVal<double> > > B;
  B.clear();
  ComputeB(sentId, this->GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, B);
  // compute the C value for this sentence
  double nLogC = ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
  double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
  double nLogP_ZGivenX = nLogC - nLogZ;
  //cerr << "nloglikelihood += " << nLogC << endl;
  // update mle for each z^*|y^* fired
  for(typename std::map< ContextType, std::map<int, LogVal<double> > >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
    const ContextType &y_ = yIter->first;
    for(std::map<int, LogVal<double> >::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double nLogb = zIter->second.s_? zIter->second.v_ : -zIter->second.v_;
      double bOverCZ = MultinomialParams::nExp(nLogb - nLogC - nLogZ);
      mle[y_][z_] += bOverCZ;
      mleMarginals[y_] += bOverCZ;
    }
  }
  return nLogP_ZGivenX;
}


#endif
