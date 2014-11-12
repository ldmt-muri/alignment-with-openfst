#!/usr/bin/env ducttape

global {
  ducttape_experimental_submitters=enable
  ducttape_experimental_imports=enable
      
  # tool paths
  brown_clusters_dir="/usr0/home/chuchenl/git/brown-cluster/"
  wammar_utils_dir="/usr0/home/chuchenl/git/alignment-with-openfst/wammar-utils"
  alignment_with_openfst_dir="/usr0/home/chuchenl/git/alignment-with-openfst/"
  featurized_hmm_pos_inducer_dir="/home/wammar/bergkirkpatrick_pos_inducer_bundle/"

  tag_dict_file=""

  # autoencoder hyperparams
  l2_strength=(L2Strength: zero=0 point_o_one=0.01 point_o_three=0.03 point_one=0.1 point_three=0.3 one=1.0 three=3.0 ten=10.0 thirty=30.0 hundred=100.0 three_hundred=300.0 )
  dirichlet_alpha=(DirichletAlpha: one_point_one=1.1 one_point_o_one=1.01 one=1.0 o_ninety_nine=0.99 o_nine=0.9 o_five=0.5 o_three=0.3 o_one=0.1 o_o_three=0.03 o_o_one=0.01 o_o_o_one=0.001 o_o_o_o_one=0.0001 )
  lbfgs_itercount=1
  em_itercount=1
  #coord_itercount=(CoordItercount: normal=50 zero=0 ten=10)
  test_with_crf_only=false
  optimize_lambdas_first=true
  min_relative_diff=(MinRelativeDiff: point_o_o_o_one=0.0001 point_o_o_one=0.001 zero=0.0)
  prefix=(Prefix: pos=pos-1 other=other other2=other2)
  #fire_precomputed_features_for_xim2=(FirePrecomputedFeaturesForXIM2: no="" yes=yes)
  fire_precomputed_features_for_xim1=(FirePrecomputedFeaturesForXIM1: no="" yes=yes)
  fire_precomputed_features_for_xip1=(FirePrecomputedFeaturesForXIP1: no="" yes=yes)
  #fire_precomputed_features_for_xip2=(FirePrecomputedFeaturesForXIP2: no="" yes=yes)
  reconstruct_brown_clusters=(ReconstructBrownClusters: yes=1 no="")
}

#import ../submitters.tape

task PreprocessData
    :: labeled_train_text=@
    :: labeled_test_text=@
    :: unlabeled_train_text=@
    :: labeled_train_labels=@
    :: labeled_test_labels=@
    :: unlabeled_train_labels=@
    > autoencoder_text
    > autoencoder_labels
    > oracle_labels
    > autoencoder_test_size
{
  cat $labeled_train_text $labeled_test_text $unlabeled_train_text > $autoencoder_text
  cat $labeled_train_labels $labeled_test_labels $unlabeled_train_labels > $oracle_labels
  cp $labeled_train_labels $autoencoder_labels
  cat $labeled_train_text $labeled_test_text | wc -l |awk -F" " '{print $1}' > $autoencoder_test_size
}

task FeaturizedHmmPosInduction
    :: wammar_utils_dir=@
    < data_file=$autoencoder_text@PreprocessData
    < gold_file=$autoencoder_labels@PreprocessData
    :: featurized_hmm_pos_inducer_dir=@
    :: l2_strength=@
    > base_conf
    > mrg_file
    > labels
    > ll
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=32g .q=shared 
{
  echo "WARNING: THIS NEEDS TO BE FIXED LATER. combine-token-label-in-mrg-file.py needs to create random labels for the mrg file when the gold labels are not provided."
  python $wammar_utils_dir/combine-token-label-in-mrg-file.py $data_file $gold_file $mrg_file
  echo "treeBank	$mrg_file" > $base_conf
  echo "numSentences	10000000" >> $base_conf
  echo "maxSentenceLength	200000" >> $base_conf
  echo "numLabels	12" >> $base_conf
  echo "iters	501" >> $base_conf
  echo "printRate	100" >> $base_conf
  echo "useStandardMultinomialMStep	false" >> $base_conf
  echo "standardMStepCountSmoothing	0.0" >> $base_conf
  echo "useGradient	false" >> $base_conf
  echo "useGlobal	false" >> $base_conf 
  echo "initialWeightsUpper	0.01" >> $base_conf
  echo "initialWeightsLower	-0.01" >> $base_conf
  echo "regularizationWeight	$l2_strength" >> $base_conf
  echo "regularizationBias	0.0" >> $base_conf
  echo "useStandardFeatures	true" >> $base_conf
  echo "lengthNGramSuffixFeature	3" >> $base_conf
  echo "useCoarsePOSFeatures	false" >> $base_conf
  echo "useBiasFeature	false" >> $base_conf
  echo "biasFeatureBias	-10.0" >> $base_conf
  echo "biasFeatureRegularizationWeight	10.0" >> $base_conf
  echo "randSeedIndex	0" >> $base_conf
  echo "monitor	true" >> $base_conf
  echo "makeThunk	false" >> $base_conf
  echo "create	true" >> $base_conf
  echo "execPoolDir	TMP" >> $base_conf 

  mkdir TMP
  
  # run
  java -cp $featurized_hmm_pos_inducer_dir/bin pos_tagging.POSInducerTester ++$base_conf 
  
  python $wammar_utils_dir/split-berg-kirkpatrick-pos-output-into-gold-vs-pred.py TMP/0.exec/guess500 gold_file_altformat $labels
  cat TMP/0.exec/log | grep "log marginal prob" | tail -n 1 > $ll 
}

task GenerateWordpairFeats 
    < data_file=$autoencoder_text@PreprocessData 
    :: brown_clusters_dir=@
    :: alignment_with_openfst_dir=@
    > wordpair_feats_file 
    > paths
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=32 .vmem=32g .q=normal 
{
  $brown_clusters_dir/wcluster --c=100 --text=$data_file --paths=$paths
  python $alignment_with_openfst_dir/parts-of-speech/create_pos_word_feats.py -b $paths -o $wordpair_feats_file
  #python $alignment_with_openfst_dir/parts-of-speech/create_code_switching_feats.py -b $paths -o $wordpair_feats_file -e $embeddings_file -s $data_file
} 

task BuildLatentCrfPosTagger
    :: alignment_with_openfst_dir=@
    > executable
   # :: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=1g .q=shared 
{
  pushd $alignment_with_openfst_dir
  #make clean -f Makefile-latentCrfPosTagger
  make -f Makefile-latentCrfPosTagger
  popd
  cp $alignment_with_openfst_dir/parts-of-speech/train-latentCrfPosTagger $executable
}

task AutoencoderPosInduction
    :: supervised=@
    < autoencoder_test_size=$autoencoder_test_size@PreprocessData
    :: labels_count=@
    :: reconstruct_brown_clusters=@
    :: wammar_utils_dir=@
    :: alignment_with_openfst_dir=@
    :: procs=$cores
    < data_file=$autoencoder_text@PreprocessData
    < oracle_labels=$oracle_labels@PreprocessData
    < gold_file=$autoencoder_labels@PreprocessData
    :: l2_strength=@
    #:: l1_strength=@
    :: dirichlet_alpha=@
    :: test_with_crf_only=@
    :: lbfgs_itercount=@
    :: em_itercount=@
    :: optimize_lambdas_first=@
    :: min_relative_diff=@
    :: prefix=@
    #:: fire_precomputed_features_for_xim2=@
    :: fire_precomputed_features_for_xim1=@
    :: fire_precomputed_features_for_xip1=@
    #:: fire_precomputed_features_for_xip2=@
    :: tag_dict_file=@
    :: labeled_test_text=@
    < executable=$executable@BuildLatentCrfPosTagger
    < wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeats 
    < tgt_brown_clusters=$paths@GenerateWordpairFeats
    # > hmm_labels 
    > autoencoder_labels 
    > out_err 
    > autoencoder_ll
    # > hmm_ll
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{
  variational="true"

  if [[ $tag_dict_file ]]; then
  python $alignment_with_openfst_dir/parts-of-speech/augment_tag_dict_with_case.py -i $tag_dict_file -o tag_dict_file.cased -t $data_file
  fi 

  python /usr0/home/chuchenl/git/cs-shared-task/sd_to_neurep.py $data_file neural /usr0/home/chuchenl/git/tweet-embedding/w2v_model --mock $oracle_labels
  test_size=$(cat $autoencoder_test_size)

  command="mpirun -np $procs $executable 
  --output-prefix $prefix
  --train-data $data_file
  --feat LABEL_BIGRAM --feat PRECOMPUTED --feat EMISSION
  --min-relative-diff $min_relative_diff
  --max-iter-count 30
  --cache-feats false
  --check-gradient true
  --optimizer adagrad --minibatch-size 8000
  --wordpair-feats $wordpair_feats_file 
  --labels-count $labels_count
  --gold-labels-filename $gold_file
  --neural-filename neural
  --test-size $test_size "

  # hide  
  #   --init-theta /usr1/home/wammar/pos-runs/italian-newcriterion/AutoencoderPosInduction/L2Strength.one+MinRelativeDiff.ze ro+OptimizeLambdasFirst.yes+Prefix.other+TestWithCrfOnly.yes-first/other.38.theta
  #  --init-lambda /usr1/home/wammar/pos-runs/italian-newcriterion/AutoencoderPosInduction/L2Strength.one+MinRelativeDiff.zero+OptimizeLambdasFirst.yes+Prefix.other+TestWithCrfOnly.yes-first/other.38.lambda
  # hide  --feat BOUNDARY_LABELS 

  if [[ $supervised ]]; then
    command="$command --supervised true"
  fi
 
  #if [[ $fire_precomputed_features_for_xim2 ]]; then
  #  command="$command --feat PRECOMPUTED_XIM2"
  #fi
 
  if [[ $fire_precomputed_features_for_xim1 ]]; then
    command="$command --feat PRECOMPUTED_XIM1"
  fi
 
  if [[ $fire_precomputed_features_for_xip1 ]]; then
    command="$command --feat PRECOMPUTED_XIP1"
  fi
 
  #if [[ $fire_precomputed_features_for_xip2 ]]; then
  #  command="$command --feat PRECOMPUTED_XIP2"
  #fi
 
  if [[ $tgt_brown_clusters && $reconstruct_brown_clusters ]]; then
    command="$command --tgt-word-classes-filename $tgt_brown_clusters"
  fi

  if [[ $l2_strength ]]; then
    command="$command --l2-strength $l2_strength"
  fi

  #if [[ $l1_strength ]]; then
  #  command="$command --l1-strength $l1_strength"
  #fi

  if [[ $dirichlet_alpha ]]; then
    command="$command --dirichlet-alpha $dirichlet_alpha"
  fi

  if [[ $variational ]]; then
    command="$command --variational-inference $variational"
  fi

  if [[ $test_with_crf_only ]]; then
    command="$command --test-with-crf-only $test_with_crf_only"
  fi

  if [[ $em_itercount ]]; then
    command="$command --max-em-iter-count $em_itercount"
  fi

  if [[ $lbfgs_itercount ]]; then
    command="$command --max-lbfgs-iter-count $lbfgs_itercount"
  fi 

  if [[ $optimize_lambdas_first ]]; then
    command="$command --optimize-lambdas-first $optimize_lambdas_first"
  fi

  if [[ $tag_dict_file ]]; then
    command="$command --tag-dict-filename tag_dict_file.cased"
  fi

  echo "executing $command..."
  $command 2> $out_err

  actual_test_size=$(wc -l $labeled_test_text |awk -F" " '{print $1}')
  echo "actual test size is $actual_test_size"  
  echo "autoencoder test size is $test_size"  

  head -n $test_size $data_file | tail -n $actual_test_size > data_file_test
  # head -n $test_size $prefix.hmm.labels > $prefix.hmm.labels.$test_size
  tail -n $actual_test_size $prefix.labels > auto_test_labels
  python $wammar_utils_dir/combine-token-label-in-one-file.py data_file_test auto_test_labels $autoencoder_labels
  # python $wammar_utils_dir/combine-token-label-in-one-file.py data_file_test $prefix.hmm.labels.$test_size $hmm_labels

  #echo "best hmm vs. latent crf:"
  #cat out_err | grep "variation of information"
  #cat out_err | grep "many-to-one"
  # cat out_err | grep "Nll" | tail -n 1 > $autoencoder_ll 
  touch $autoencoder_ll
  # cat out_err | grep "global max loglikelihood is" | tail -n 1 > $hmm_ll
}

task Evaluate
    :: test_size=@
    :: wammar_utils_dir=@
    :: labeled_test_text=@
    :: labeled_test_labels=@
    < labels=(Model:
                featurized_hmm=$labels@FeaturizedHmmPosInduction
                autoencoder=$autoencoder_labels@AutoencoderPosInduction)
                # hmm=$hmm_labels@AutoencoderPosInduction)
    < ll=(Model:
                featurized_hmm=$ll@FeaturizedHmmPosInduction
                autoencoder=$autoencoder_ll@AutoencoderPosInduction)
                # hmm=$hmm_ll@AutoencoderPosInduction)
    > gold_file2
    > scores 
{
  # convert tokens and gold labels to token/label format
  python $wammar_utils_dir/combine-token-label-in-one-file.py $labeled_test_text $labeled_test_labels $gold_file2
  python $wammar_utils_dir/score-classes.py $gold_file2 $labels 2> $scores
  python $wammar_utils_dir/score-vm.py $gold_file2 $labels 2>> $scores
  cat $ll >> $scores
}
