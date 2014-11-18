#!/usr/bin/env ducttape

global {
  ducttape_experimental_submitters=enable
  ducttape_experimental_imports=enable
      
  # tool paths
  brown_clusters_dir="/usr0/home/wammar/git/brown-cluster/"
  wammar_utils_dir="/usr0/home/wammar/alignment-with-openfst/wammar-utils"
  alignment_with_openfst_dir="/usr0/home/wammar/alignment-with-openfst/"
  featurized_hmm_pos_inducer_dir="/home/wammar/bergkirkpatrick_pos_inducer_bundle/"

  # autoencoder hyperparams
  l2_strength=(L2Strength: point_three=0.3 zero=0 point_o_one=0.01 point_o_three=0.03 point_one=0.1  one=1.0 two_point_five=2.5)
  dirichlet_alpha=(DirichletAlpha: one_point_one=1.1 one_point_five=1.5 one_point_o_one=1.01 one=1.0 point_ninety_nine=0.99 point_nine=0.9 point_five=0.5 point_three=0.3 point_one=0.1 point_o_three=0.03 point_o_one=0.01 point_o_o_one=0.001 point_o_o_o_one=0.0001 )
  lbfgs_itercount=1
  em_itercount=1
  coord_itercount=(CoordIterCount: fifty=50 seventy=70 twenty=20 hundred=100 thousand=1000 zero=0 ten=10)
  test_with_crf_only=""
  optimize_lambdas_first=true
  min_relative_diff=0.0001
  prefix=(Prefix: pos=pos3 other=other other2=other2 exp=exp)
  labels_count=12
  feature_set=(FeatureSet: full="full" hk="hk" basic="basic")
  effective_unlabeled_lines=(EffectiveUnlabeledLines:
                infinity=10000000
                0k=0
                1k=1000
                2k=2000
                4k=4000
                8k=8000
                16k=16000)

}

#import ../submitters.tape

task PreprocessData
    :: labeled_train_text=@
    :: labeled_test_text=@
    :: unlabeled_train_text=@
    :: effective_unlabeled_lines=@
    :: labeled_train_labels=@
    :: labeled_test_labels=@
    > autoencoder_text
    > autoencoder_labels
    > autoencoder_test_size
    > fhmm_text
    > fhmm_labels
    > fhmm_test_size    
{
  head -n $effective_unlabeled_lines $unlabeled_train_text > effective_unlabeled_train_text

  # prepare input for CRF autoencoder.
  cat $labeled_train_text $labeled_test_text effective_unlabeled_train_text > $autoencoder_text
  cp $labeled_train_labels $autoencoder_labels
  cat $labeled_train_text $labeled_test_text | wc -l |awk -F" " '{print $1}' > $autoencoder_test_size

  # prepare input for feature rich HMM. 
  cat $labeled_test_text effective_unlabeled_train_text > $fhmm_text
  cat $labeled_test_labels effective_unlabeled_train_text > $fhmm_labels
  cat $labeled_test_text | wc -l |awk -F" " '{print $1}' > $fhmm_test_size
}

task FeaturizedHmmPosInduction
    :: wammar_utils_dir=@
    < data_file=$fhmm_text@PreprocessData
    < gold_file=$fhmm_labels@PreprocessData
    < test_size=$fhmm_test_size@PreprocessData
    :: featurized_hmm_pos_inducer_dir=@
    :: l2_strength=@
    > base_conf
    > mrg_file
    > labels
    > ll
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=32g .q=shared 
{
  echo "WARNING: THIS NEEDS TO BE FIXED. combine-token-label-in-mrg-file.py needs to create random labels for the mrg file when the gold labels are not provided."
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

  # first, separate gold vs. predictions   
  python $wammar_utils_dir/split-berg-kirkpatrick-pos-output-into-gold-vs-pred.py TMP/0.exec/guess500 gold_file_altformat all_labels

  # then, all_labels consists of a number of test-set predictions followed by a number of unlabeled-train-set predictions. 
  # we want to evaluate only the first part.
  head -n "$(cat $test_size)" all_labels > $labels

  cat TMP/0.exec/log | grep "log marginal prob" | tail -n 1 > $ll 
}

task GenerateWordpairFeats 
    :: precomputed_brown_clusters=@
    < data_file=$autoencoder_text@PreprocessData
    :: brown_clusters_dir=@
    :: alignment_with_openfst_dir=@
    :: feature_set=@
    > wordpair_feats_file 
    > paths
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=32 .vmem=32g .q=normal 
{
  $brown_clusters_dir/wcluster --c=100 --text=$data_file --paths=temp_paths
  if [[ $precomputed_brown_clusters ]]; then
    python $alignment_with_openfst_dir/parts-of-speech/remove_irrelevant_words.py -b $precomputed_brown_clusters -r temp_paths -o $paths
    cp $precomputed_brown_clusters $paths
  else
    mv temp_paths $paths
  fi

  ## the precomputed features depend on the feature set 
  ## "hk" is the feature set used in Haghighi & Klein 2006
  if [[ $feature_set == "hk" ]]; then
    python $alignment_with_openfst_dir/parts-of-speech/create_pos_word_feats.py -b $paths -o $wordpair_feats_file -hk
  fi

  ## full = the full feature set used in Ammar et al. 2014
  if [[ $feature_set == "full" ]]; then
    python $alignment_with_openfst_dir/parts-of-speech/create_pos_word_feats.py -b $paths -o $wordpair_feats_file
  fi

  ## basic = emissions and transitions only. don't generate anything here
  if [[ $feature_set == "basic" ]]; then
    touch $wordpair_feats_file
  fi
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
    :: init_theta=@
    :: init_lambda=@
    :: supervised=@
    < autoencoder_test_size=$autoencoder_test_size@PreprocessData
    :: labels_count=@
    :: reconstruct_brown_clusters=@
    :: wammar_utils_dir=@
    :: alignment_with_openfst_dir=@
    :: procs=$cores
    < data_file=$autoencoder_text@PreprocessData
    < gold_file=$autoencoder_labels@PreprocessData
    :: l2_strength=@
    :: dirichlet_alpha=@
    :: test_with_crf_only=@
    :: lbfgs_itercount=@
    :: em_itercount=@
    :: optimize_lambdas_first=@
    :: min_relative_diff=@
    :: prefix=@
    :: tag_dict_file=@
    :: labeled_test_text=@
    :: feature_set=@
    :: coord_itercount=@
    < executable=$executable@BuildLatentCrfPosTagger
    < wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeats 
    < tgt_brown_clusters=$paths@GenerateWordpairFeats
    > hmm_labels 
    > autoencoder_labels 
    > out_err 
    > autoencoder_ll
    # > hmm_ll
    > auto_test_labels
    > hmm_test_labels
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{
  variational="true"

  if [[ $tag_dict_file ]]; then
  python $alignment_with_openfst_dir/parts-of-speech/augment_tag_dict_with_case.py -i $tag_dict_file -o tag_dict_file.cased -t $data_file
  fi 

  test_size=$(cat $autoencoder_test_size)
  total_l2=$(echo "$l2_strength * $test_size" | bc -l)
  echo "effective l2 strength = $total_l2"
  
  command="nice -10 mpirun -np $procs $executable 
  --output-prefix $prefix
  --train-data $data_file
  --feat LABEL_BIGRAM
  --min-relative-diff $min_relative_diff
  --max-iter-count $coord_itercount
  --cache-feats false
  --check-gradient false
  --optimizer lbfgs --minibatch-size 100000
  #--optimizer sgd
  --wordpair-feats $wordpair_feats_file 
  --labels-count $labels_count
  --gold-labels-filename $gold_file"

  # initial values of theta 
  if [[ $init_theta ]]; then
    command="$command --init-theta $init_theta"
  fi

  # initial values of lambda 
  if [[ $init_lambda ]]; then
    command="$command --init-lambda $init_lambda"
  fi

  # specify feature templates in each feature set.
  if [[ $feature_set == "basic" ]]; then
    command="$command --feat EMISSION"
  fi
  if [[ $feature_set == "hk" ]]; then
    command="$command --feat EMISSION --feat PRECOMPUTED"
  fi
  if [[ $feature_set == "full" ]]; then
    command="$command --feat PRECOMPUTED --feat PRECOMPUTED_XIM1  --feat PRECOMPUTED_XIP1"
  fi

  if [[ $supervised ]]; then
    command="$command --supervised true"
  fi
 
  if [[ $tgt_brown_clusters && $reconstruct_brown_clusters ]]; then
    command="$command --tgt-word-classes-filename $tgt_brown_clusters"
  fi

  if [[ $l2_strength ]]; then
    command="$command --l2-strength $total_l2"
  fi

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

  if [[ $init_theta ]]; then
    cp $prefix.final.labels $prefix.hmm.labels
  fi


  if [[ $supervised ]]; then
    tail -n $actual_test_size $prefix.supervised.labels > $auto_test_labels    
  else
    head -n $test_size $prefix.final.labels | tail -n $actual_test_size > $auto_test_labels
    head -n $test_size $prefix.hmm.labels   | tail -n $actual_test_size > $hmm_test_labels
  fi
 
  python $wammar_utils_dir/combine-token-label-in-one-file.py data_file_test $auto_test_labels $autoencoder_labels
  python $wammar_utils_dir/combine-token-label-in-one-file.py data_file_test $hmm_test_labels $hmm_labels
  
  touch $autoencoder_ll
}

task Evaluate
    :: wammar_utils_dir=@
    :: labeled_test_text=@
    :: labeled_test_labels=@
    < labels=(Model:
                autoencoder=$autoencoder_labels@AutoencoderPosInduction
                featurized_hmm=$labels@FeaturizedHmmPosInduction
                hmm=$hmm_labels@AutoencoderPosInduction)
    > gold_file
    > scores 
{
  # convert tokens and gold labels to token/label format
  python $wammar_utils_dir/combine-token-label-in-one-file.py $labeled_test_text $labeled_test_labels $gold_file
  python $wammar_utils_dir/score-classes.py $gold_file $labels 2> $scores
  python $wammar_utils_dir/score-vm.py $gold_file $labels 2>> $scores
  #cat $ll >> $scores
}
