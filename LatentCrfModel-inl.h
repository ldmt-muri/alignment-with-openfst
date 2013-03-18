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
      cerr << abs((mleMarginals[y_] - unnormalizedMarginalProbz_giveny_) / mleMarginals[y_]); 
      cerr << "mleMarginals[y_] = " << mleMarginals[y_] << " unnormalizedMarginalProbz_giveny_ = " << unnormalizedMarginalProbz_giveny_;
      cerr << " --error ignored, but try to figure out what's wrong!" << endl;
    }
    // normalize the mle estimates to sum to one for each context
    for(MultinomialParams::MultinomialParam::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double normalizedProbz_giveny_ = zIter->second / mleMarginals[y_];
      mle[y_][z_] = normalizedProbz_giveny_;
      // take the nlog
      mle[y_][z_] = MultinomialParams::nLog(mle[y_][z_]);
    }
  }
}

template <typename ContextType>
void UpdateThetaMleForSent(const unsigned sentId, 
			   MultinomialParams::ConditionalMultinomialParam<ContextType> &mle, 
			   std::map<ContextType, double> &mleMarginals) {
  if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
    std::cerr << "sentId = " << sentId << endl;
  }
  assert(sentId < data.size());
  // build the FST
  fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
  std::vector<FstUtils::LogWeight> alphas, betas;
  BuildThetaLambdaFst(sentId, data[sentId], thetaLambdaFst, alphas, betas);
  // compute the B matrix for this sentence
  std::map< ContextType, std::map< int, LogVal<double> > > B;
  B.clear();
  ComputeB(sentId, this->data[sentId], thetaLambdaFst, alphas, betas, B);
  // compute the C value for this sentence
  double nLogC = ComputeNLogC(thetaLambdaFst, betas);
  //cerr << "nloglikelihood += " << nLogC << endl;
  // update mle for each z^*|y^* fired
  for(typename std::map< ContextType, std::map<int, LogVal<double> > >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
    const ContextType &y_ = yIter->first;
    for(std::map<int, LogVal<double> >::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double nLogb = zIter->second.s_? zIter->second.v_ : -zIter->second.v_;
      double bOverC = MultinomialParams::nExp(nLogb - nLogC);
      mle[y_][z_] += bOverC;
      mleMarginals[y_] += bOverC;
    }
  }
}


#endif
