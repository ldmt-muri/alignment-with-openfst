#ifndef _I_ALIGNMENT_MODEL_H_
#define _I_ALIGNMENT_MODEL_H_

#include <vector>
#include <string>

class IAlignmentModel
{
 public:
  
  // print current model paramters
  virtual void PrintParams() = 0;
  
  // persist current model parameters
  virtual void PersistParams(const std::string& outputFilename) = 0;
  
  // use the training data provided at instantiation to optimize model parameters
  virtual void Train() = 0;
  
  // use the current model paramters to align training data
  virtual void Align() = 0;
  
  // use the current model parameters to align a test set
  virtual void AlignTestSet(const std::string &srcTestSetFilename, const std::string &tgtTestSetFilename, const std::string &alignmentsFilename) = 0;
  
};

#endif
