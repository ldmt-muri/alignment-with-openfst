#ifndef _COMPARE_CLUSTERS_H_
#define _COMPARE_CLUSTERS_H_

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <assert.h>
#include <vector>
#include <set>

// this class implements the variation of information metric for comparing two clusterings of the same data points.
// details can be found at http://www.cs.cmu.edu/~wammar/www.stat.washington.edu/mmp/Papers/jasa-compare.ps
class ClustersComparer {
  ClustersComparer() {    
  }

  // assumptions:
  // - confusionMatrix is empty
  static void BuildConfusionMatrix(const std::vector<std::string> &a, const std::vector<std::string> &b, 
				   std::map<std::string, std::map<std::string, unsigned> > &confusionMatrix) {
    assert(confusionMatrix.size() == 0);

    // initialize the confusion matrix to all zeros
    std::set<std::string> aClusters, bClusters;
    for(int i = 0; i < a.size(); i++) {
      aClusters.insert(a[i]);
      bClusters.insert(b[i]);
    }
    for(std::set<std::string>::const_iterator aClustersIter = aClusters.begin();
	aClustersIter != aClusters.end();
	aClustersIter++) {
      for(std::set<std::string>::const_iterator bClustersIter = bClusters.begin();
	  bClustersIter != bClusters.end();
	  bClustersIter++) {
	confusionMatrix[*aClustersIter][*bClustersIter] = 0;
      }
    }

    // fill in the entries
    for(int i = 0; i < a.size(); i++) {
      const std::string &aCluster = a[i], &bCluster = b[i];
      confusionMatrix[aCluster][bCluster]++;
    }
  }

  static string PrintConfusionMatrix(const std::map<std::string, std::map<std::string, unsigned> > &confusionMatrix) {
    assert(confusionMatrix.size() != 0);
    stringstream ss;
    // print headers (i.e. b clusters)
    ss << "\t";
    for(std::map<std::string, unsigned>::const_iterator bIter = confusionMatrix.begin()->second.begin(); 
	bIter != confusionMatrix.begin()->second.end();
	bIter++) {
      ss << bIter->first << "\t";
    }
    ss << endl << endl;
    for(std::map<std::string, std::map<std::string, unsigned> >::const_iterator aIter = confusionMatrix.begin();
	aIter != confusionMatrix.end();
	aIter++) {
      // print column followed by values
      ss << aIter->first << "\t";
      for(std::map<std::string, unsigned>::const_iterator bIter = aIter->second.begin(); 
	  bIter != aIter->second.end();
	  bIter++) {
	ss << bIter->second << "\t";
      }
      ss << endl << endl;
    }
    return ss.str();
  }

  // assumtpions:
  // - clusterSizes is empty
  static void ComputeClusterSizes(const std::vector<std::string> &clustering,  
				  std::map<std::string, unsigned> &clusterSizes) {
    assert(clusterSizes.size() == 0);
    for(std::vector<std::string>::const_iterator aIter = clustering.begin(); aIter != clustering.end(); aIter++) {
      clusterSizes[*aIter]++;
    }
  }

  static double ComputeClusterEntropies(const std::map<std::string, unsigned> &clusterSizes, unsigned dataSize) {
    assert(dataSize != 0);

    double entropy = 0.0;
    for(std::map<std::string, unsigned>::const_iterator sizeIter = clusterSizes.begin(); sizeIter != clusterSizes.end(); sizeIter++) {
      entropy -= ((double)sizeIter->second / dataSize) * log((double)sizeIter->second / dataSize);
    }

    return entropy;
  }

  static bool VerifyTwoClusteringsAreValid(const std::vector<std::string> &a, const std::vector<std::string> &b) {
    bool valid = true;
    // make sure both a and b are clusterings of the same set of data points.
    if(a.size() != b.size() || a.size() == 0) {
      valid = false;
    }
    return valid;
  }
  
  static double ComputeMutualInformation(const std::map<std::string, std::map<std::string, unsigned> > confusionMatrix,
					 const std::map<std::string, unsigned> aCounts, 
					 const std::map<std::string, unsigned> bCounts,
					 int dataSize) {
    assert(dataSize != 0);
    double mutualInformation = 0.0;
    for(std::map<std::string, unsigned>::const_iterator aIter = aCounts.begin(); aIter != aCounts.end(); aIter++) {
      for(std::map<std::string, unsigned>::const_iterator bIter = bCounts.begin(); bIter != bCounts.end(); bIter++) {
	unsigned intersectionSize = confusionMatrix.find(aIter->first)->second.find(bIter->first)->second;
	if(intersectionSize == 0) {
	  continue;
	}		  
	double term = (double) intersectionSize / dataSize;
	term *= log(term / ((double)aIter->second/dataSize * bIter->second/dataSize));
	assert(!std::isnan(term));
	mutualInformation += term;
      }
    }
    return mutualInformation;
  }

 public:
  // a and b are the two clusterings.
  // keys of a and b are identical, representing the data points.
  // a[point] and b[point] are the classes to which 'point' belongs in clustering a and b (respectively).
  static double ComputeVariationOfInformation(const std::vector<std::string> &a, const std::vector<std::string> &b) {
    assert(VerifyTwoClusteringsAreValid(a, b));
    std::map<std::string, std::map<std::string, unsigned> > confusionMatrix;
    BuildConfusionMatrix(a, b, confusionMatrix);
    cerr << "confusion matrix:" << endl << PrintConfusionMatrix(confusionMatrix) << endl;
    std::map<std::string, unsigned> aCounts, bCounts;
    ComputeClusterSizes(a, aCounts);
    ComputeClusterSizes(b, bCounts);
    double aEntropy = ComputeClusterEntropies(aCounts, a.size());
    cerr << "H(A) = " << aEntropy << endl;
    double bEntropy = ComputeClusterEntropies(bCounts, b.size());
    cerr << "H(B) = " << bEntropy << endl;
    double mi = ComputeMutualInformation(confusionMatrix, aCounts, bCounts, a.size());
    cerr << "MI(A,B) = " << mi << endl;
    double vi = aEntropy - mi + bEntropy - mi;
    cerr << "VI = " << vi << endl;
    double ht = bEntropy, hc = aEntropy, ht2c = hc - mi, hc2t = ht - mi, h = 1 - hc2t / ht, c = 1 - ht2c / hc, vm = 2 * h * c / (h + c);
    cerr << "VMeasure = " << vm << endl;
    return vi;
  }

  // assumes b is the gold/reference
  static double ComputeManyToOne(const std::vector<std::string> &a, const std::vector<std::string> &b) {
    assert(VerifyTwoClusteringsAreValid(a, b));
    std::map<std::string, std::map<std::string, unsigned> > confusionMatrix;
    BuildConfusionMatrix(a, b, confusionMatrix);
    unsigned correct = 0;
    for(std::map<std::string, std::map<std::string, unsigned> >::const_iterator aIter = confusionMatrix.begin(); 
	aIter != confusionMatrix.end(); 
	aIter++) {
      unsigned max = 0;
      for(std::map<std::string, unsigned>::const_iterator bIter = aIter->second.begin();
	  bIter != aIter->second.end();
	  bIter++) {
	if(bIter->second > max) {
	  max = bIter->second;
	}
      }
      correct += max;
    }
    return 1.0 * correct / a.size();
  }
};

#endif
