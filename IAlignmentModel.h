#ifndef _I_ALIGNMENT_MODEL_H_
#define _I_ALIGNMENT_MODEL_H_

#include <vector>
#include <string>

class IAlignmentSampler {
  public:

    // print current model paramters
    virtual void PrintParams() = 0;
    
    // persist current model parameters
    virtual void PersistParams(const std::string& outputFilename) = 0;

    // use the training data provided at instantiation to optimize model parameters
    virtual void Train();

    // use the current model paramters to align training data
    virtual void Align();

    // use the current model parameters to align a test set
    virtual void AlignTestSet(const string &srcTestSetFilename, const string &tgtTestSetFilename, const string &alignmentsFilename);

};

#endif
