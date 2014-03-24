#ifndef _UNSUPERVISED_SEQUENCE_TAGGING_MODEL_H_
#define _UNSUPERVISED_SEQUENCE_TAGGING_MODEL_H_

#include <vector>
#include <string>

#include "../wammar-utils/FstUtils.h"
#include "../wammar-utils/ClustersComparer.h"
#include "../wammar-utils/StringUtils.h"
#include "VocabEncoder.h"
#include "LearningInfo.h"

using namespace std;

class UnsupervisedSequenceTaggingModel {

 public:

  virtual void Train() = 0;

  virtual void Label(vector<int64_t> &tokens, vector<int> &labels) = 0;

  void Label(vector<string> &tokens, vector<int> &labels) {
    assert(labels.size() == 0);
    assert(tokens.size() > 0);
    vector<int64_t> tokensInt;
    for(unsigned i = 0; i < tokens.size(); i++) {
      tokensInt.push_back(vocabEncoder.Encode(tokens[i]));
    }
    Label(tokensInt, labels);
  }

  void Label(vector<vector<int64_t> > &tokens, vector<vector<int> > &labels) {
    assert(labels.size() == 0);
    labels.resize(tokens.size());
    for(unsigned i = 0; i < tokens.size(); i++) {
      Label(tokens[i], labels[i]);
    }
  }

  void Label(vector<vector<string> > &tokens, vector<vector<int> > &labels) {
    assert(labels.size() == 0);
    labels.resize(tokens.size());
    for(unsigned i = 0 ; i <tokens.size(); i++) {
      Label(tokens[i], labels[i]);
    }
  }

  void Label(string &inputFilename, string &outputFilename) {
    std::vector<std::vector<std::string> > tokens;
    StringUtils::ReadTokens(inputFilename, tokens);
    vector<vector<int> > labels;
    Label(tokens, labels);
    StringUtils::WriteTokens(outputFilename, labels);
  }
  
  // evaluate
  double ComputeVariationOfInformation(string &aLabelsFilename, string &bLabelsFilename) {
    vector<string> clusteringA, clusteringB;
    vector<vector<string> > clusteringAByLine, clusteringBByLine;
    StringUtils::ReadTokens(aLabelsFilename, clusteringAByLine);
    StringUtils::ReadTokens(bLabelsFilename, clusteringBByLine);
    if(clusteringAByLine.size() != clusteringBByLine.size()) {
      cerr << "ERROR COMPUTING VARIATION OF INFORMATION." << endl;
      cerr << "clusteringAByLine.size() = " << clusteringAByLine.size() << endl;
      cerr << "clusteringBByLine.size() = " << clusteringBByLine.size() << endl;
      assert(false);
    }
    for(unsigned i = 0; i < clusteringAByLine.size(); i++) {
      assert(clusteringAByLine[i].size() == clusteringBByLine[i].size());
      for(unsigned j = 0; j < clusteringAByLine[i].size(); j++) {
        clusteringA.push_back(clusteringAByLine[i][j]);
        clusteringB.push_back(clusteringBByLine[i][j]);			    
      }
    }
    return ClustersComparer::ComputeVariationOfInformation(clusteringA, clusteringB);
  }
  
  double ComputeManyToOne(string &aLabelsFilename, string &bLabelsFilename) {
    vector<string> clusteringA, clusteringB;
    vector<vector<string> > clusteringAByLine, clusteringBByLine;
    StringUtils::ReadTokens(aLabelsFilename, clusteringAByLine);
    StringUtils::ReadTokens(bLabelsFilename, clusteringBByLine);
    assert(clusteringAByLine.size() == clusteringBByLine.size());
    for(unsigned i = 0; i < clusteringAByLine.size(); i++) {
      assert(clusteringAByLine[i].size() == clusteringBByLine[i].size());
      for(unsigned j = 0; j < clusteringAByLine[i].size(); j++) {
        clusteringA.push_back(clusteringAByLine[i][j]);
        clusteringB.push_back(clusteringBByLine[i][j]);			    
      }
    }
    return ClustersComparer::ComputeManyToOne(clusteringA, clusteringB);
  }

 protected:
  
 UnsupervisedSequenceTaggingModel(const string &vocabEncoderFilenameInitializer, 
                                  LearningInfo &learningInfo) : 
  vocabEncoder(vocabEncoderFilenameInitializer, learningInfo),
    learningInfo(learningInfo)
  {
  }

  ~UnsupervisedSequenceTaggingModel() { }
  
 public:
  // vocab encoders
  VocabEncoder vocabEncoder;
  LearningInfo &learningInfo;
  
};

#endif
