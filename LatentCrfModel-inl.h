#ifndef _LATENT_CRF_MODEL_INL_H_
#define _LATENT_CRF_MODEL_INL_H_

template <typename ContextType>
void ComputeNllZGivenXThetaGradient(MultinomialParams::ConditionalMultinomialParam<ContextType> &gradient) {
  assert(learningInfo.thetaOptMethod->algorithm == OptAlgorithm::GRADIENT_DESCENT);

  double Nll = 0;
  
  for(int sentId = 0; sentId < examplesCount; sentId++) {
    
    // build the FSTs
    fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
    vector<FstUtils::LogWeight> thetaLambdaAlphas, thetaLambdaBetas;
    BuildThetaLambdaFst(sentId, GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);

    // compute the B matrix for this sentence
    std::map< ContextType, std::map< int, LogVal<double> > > B;
    B.clear();
    ComputeB(sentId, this->GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, B);
    
    // compute the C value for this sentence
    double nLogC = ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);    

    // update the gradient for every theta used in this sentence
    for(typename std::map< ContextType, std::map<int, LogVal<double> > >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
      int context = GetContextOfTheta(sentId, yIter->first);
      for(std::map<int, LogVal<double> >::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
	int z_ = zIter->first;
	double nLogb = -log<double>(zIter->second);
	assert(zIter->second.s_ == false); //  all B values are supposed to be positive
	double nlogTheta_ = GetNLogTheta(context, z_);
	double nlogGradientUpdate = nLogb - nLogC - nlogTheta_;
	double gradientUpdate = MultinomialParams::nExp(nlogGradientUpdate);
	gradient[context][z_] -= gradientUpdate;
	
      } // end of loop over context
    } // end of gradient updates
  } // end of foreach sent

  /*
  // add the penalty term for all thetas
  double delta = 1;
  typename std::map<ContextType, double> thetaMarginals;
  for(typename std::map< ContextType, std::map<int, double> >::const_iterator yIter = nLogThetaGivenOneLabel.params.begin(); 
      yIter != nLogThetaGivenOneLabel.params.end(); 
      ++yIter) {
    // first compute the marginal for each context
    thetaMarginals[yIter->first] = 0.0;
    for(std::map<int, double>::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); ++zIter) {
      thetaMarginals[yIter->first] += MultinomialParams::nExp(nLogThetaGivenOneLabel[yIter->first][zIter->first]);
    }
    // then update the gradient for all events
    for(std::map<int, double>::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); ++zIter) {
      gradient[yIter->first][zIter->first] += 2.0 / delta * sin( (1.0 - thetaMarginals[yIter->first]) / delta);
    }
  } 
  */ 
} // end of ComputeNllZGivenXAndLambdaGradient


#endif
