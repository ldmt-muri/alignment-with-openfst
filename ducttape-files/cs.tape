#!/usr/bin/env ducttape

global {
  ducttape_experimental_submitters=enable
  ducttape_experimental_imports=enable
      
  # tool paths
   brown_clusters_dir="/usr0/home/chuchenl/git/brown-cluster/"
   wammar_utils_dir="/usr0/home/chuchenl/git/alignment-with-openfst/wammar-utils"
   alignment_with_openfst_dir="/usr0/home/chuchenl/git/alignment-with-openfst/"
   featurized_hmm_pos_inducer_dir="/home/wammar/bergkirkpatrick_pos_inducer_bundle/"
   evaluation_script_dir="/usr0/home/chuchenl/git/cs-shared-task/Evaluation/Scripts/"
   cs_sharedtask_dir="/usr0/home/chuchenl/git/cs-shared-task"


  tag_dict_file=""

  # autoencoder hyperparams
  l2_strength=(L2Strength: zero=0 point_o_one=0.01 point_o_three=0.03 point_one=0.1 point_three=0.3 one=1.0 three=3.0 ten=10.0 thirty=30.0 hundred=100.0 three_hundred=300.0 )
  dirichlet_alpha=(DirichletAlpha: one_point_one=1.1 one_point_o_one=1.01 one=1.0 o_ninety_nine=0.99 o_nine=0.9 o_five=0.5 o_three=0.3 o_one=0.1 o_o_three=0.03 o_o_one=0.01 o_o_o_one=0.001 o_o_o_o_one=0.0001 )
  em_itercount=1
  #coord_itercount=(CoordItercount: normal=50 zero=0 ten=10)
  test_with_crf_only=true
  optimize_lambdas_first=true
  min_relative_diff=(MinRelativeDiff: point_o_o_o_one=0.0001 point_o_o_one=0.001 zero=0.0)
  prefix="cs-chucheng"
  #fire_precomputed_features_for_xim2=(FirePrecomputedFeaturesForXIM2: no="" yes=yes)
  fire_precomputed_features_for_xim1=(FirePrecomputedFeaturesForXIM1: no="" yes=yes)
  fire_precomputed_features_for_xip1=(FirePrecomputedFeaturesForXIP1: no="" yes=yes)
  #fire_precomputed_features_for_xip2=(FirePrecomputedFeaturesForXIP2: no="" yes=yes)
  reconstruct_brown_clusters=(ReconstructBrownClusters: yes=1 no="")
  is_crfsuite=(Model: autoencoder="" crfsuite="--crfsuite")
  embeddings=(UseEmbeddings: yes="-e" no="")
}

#import ../submitters.tape

task PreprocessData
    :: labeled_train_text=@
    :: labeled_test_text=@
    :: unlabeled_train_text=@
    :: labeled_train_labels=@
    :: labeled_test_labels=@
    :: tsv_template=@
    :: cs_sharedtask_dir=@
    > autoencoder_text
    > autoencoder_labels
    > autoencoder_test_size
    > gold_tsv
    > pred_ldig
    > pred_cld2
{
  cat $labeled_train_text $labeled_test_text $unlabeled_train_text > $autoencoder_text
  cp $labeled_train_labels $autoencoder_labels
  cat $labeled_train_text $labeled_test_text | wc -l |awk -F" " '{print $1}' > $autoencoder_test_size
  python $cs_sharedtask_dir/crf_labels_to_tsv.py $labeled_test_labels $tsv_template $gold_tsv

  python $cs_sharedtask_dir/get_ldig.py $autoencoder_text $pred_ldig 
  python $cs_sharedtask_dir/get_cld2.py $autoencoder_text $pred_cld2 
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
    :: embeddings=@
    :: embeddings_file=@
    :: brown_clusters_dir=@
    :: alignment_with_openfst_dir=@
    :: labeled_train_text=@
    :: labeled_train_labels=@
    :: word_list=@
    > wordpair_feats_file 
    > paths
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=32 .vmem=32g .q=normal 
{
  $brown_clusters_dir/wcluster --c=100 --text=$data_file --paths=$paths
  # python $alignment_with_openfst_dir/parts-of-speech/create_pos_word_feats.py -b $paths -o $wordpair_feats_file
  python $alignment_with_openfst_dir/parts-of-speech/create_code_switching_feats.py -b $paths -o $wordpair_feats_file -s $data_file -t $labeled_train_text -l $labeled_train_labels -w $word_list $embeddings $embeddings_file
} 

task GenerateCRFSuiteInput
< wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeats
:: other_pred_arg="--other-predictors"
:: labeled_train_text=@
:: labeled_train_labels=@
:: labeled_test_text=@
:: labeled_test_labels=@
:: cs_sharedtask_dir=@
> crfsuite_train
> crfsuite_test
{
  python $cs_sharedtask_dir/get_ldig.py $labeled_train_text ldig 
  python $cs_sharedtask_dir/get_cld2.py $labeled_test_text cld2 
  python $cs_sharedtask_dir/converters/autoencoder_to_crfsuite.py $labeled_train_text $labeled_train_labels $wordpair_feats_file $crfsuite_train --pre 2 --post 2 $other_pred_arg ldig cld2
  python $cs_sharedtask_dir/converters/autoencoder_to_crfsuite.py $labeled_test_text $labeled_test_labels $wordpair_feats_file $crfsuite_test --pre 2 --post 2 $other_pred_arg ldig cld2
}

task TrainCRFSuite
< train=$crfsuite_train@GenerateCRFSuiteInput
> model
{
  crfsuite learn -m $model $train
}

task BuildLatentCrfPosTagger
    :: alignment_with_openfst_dir=@
    > executable
   # :: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=1g .q=shared 
{
  #source ~/.profile
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
    :: fire_precomputed_features_for_xim2=@
    :: fire_precomputed_features_for_xim1=@
    :: fire_precomputed_features_for_xip1=@
    :: fire_precomputed_features_for_xip2=@
    :: tag_dict_file=@
    :: labeled_test_text=@
    < executable=$executable@BuildLatentCrfPosTagger
    < wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeats 
    < tgt_brown_clusters=$paths@GenerateWordpairFeats
    > autoencoder_labels 
    > out_err 
    > auto_test_labels
    < pred_ldig=$pred_ldig@PreprocessData
    < pred_cld2=$pred_cld2@PreprocessData
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{
  variational="true"

  if [[ $tag_dict_file ]]; then
  python $alignment_with_openfst_dir/parts-of-speech/augment_tag_dict_with_case.py -i $tag_dict_file -o tag_dict_file.cased -t $data_file
  fi 

  test_size=$(cat $autoencoder_test_size)

  command="mpirun -np $procs $executable 
  --output-prefix $prefix
  --train-data $data_file
  --feat LABEL_BIGRAM --feat PRECOMPUTED --feat EMISSION
  --min-relative-diff $min_relative_diff
  --max-iter-count 30
  --cache-feats false
  --cache-gradient true
  --optimizer adagrad --minibatch-size 800
  --wordpair-feats $wordpair_feats_file 
  --labels-count $labels_count
  --gold-labels-filename $gold_file
  --test-size $test_size 
  --feat OTHER_ALIGNERS
  --other-aligners-output-filenames $pred_ldig $pred_cld2"

  # hide  
  #   --init-theta /usr1/home/wammar/pos-runs/italian-newcriterion/AutoencoderPosInduction/L2Strength.one+MinRelativeDiff.ze ro+OptimizeLambdasFirst.yes+Prefix.other+TestWithCrfOnly.yes-first/other.38.theta
  #  --init-lambda /usr1/home/wammar/pos-runs/italian-newcriterion/AutoencoderPosInduction/L2Strength.one+MinRelativeDiff.zero+OptimizeLambdasFirst.yes+Prefix.other+TestWithCrfOnly.yes-first/other.38.lambda
  # hide  --feat BOUNDARY_LABELS 

  if [[ $supervised ]]; then
    command="$command --supervised true"
  fi
 
  if [[ $fire_precomputed_features_for_xim2 ]]; then
    command="$command --feat PRECOMPUTED_XIM2"
  fi
 
  if [[ $fire_precomputed_features_for_xim1 ]]; then
    command="$command --feat PRECOMPUTED_XIM1"
  fi
 
  if [[ $fire_precomputed_features_for_xip1 ]]; then
    command="$command --feat PRECOMPUTED_XIP1"
  fi
 
  if [[ $fire_precomputed_features_for_xip2 ]]; then
    command="$command --feat PRECOMPUTED_XIP2"
  fi
 
  if [[ $tgt_brown_clusters && $reconstruct_brown_clusters ]]; then
    command="$command --tgt-word-classes-filename $tgt_brown_clusters"
  fi

  if [[ $l2_strength ]]; then
    # command="$command --l2-strength $l2_strength"
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
  tail -n $actual_test_size $prefix.labels > $auto_test_labels
  python $wammar_utils_dir/combine-token-label-in-one-file.py data_file_test $auto_test_labels $autoencoder_labels
}

task TestCRFSuite
  < test=$crfsuite_test@GenerateCRFSuiteInput
  < model=$model@TrainCRFSuite
  :: tsv_template=@
  > test_results
{
  crfsuite tag -m $model -t $test > $test_results
}

task Evaluate
    < gold_tsv=$gold_tsv@PreprocessData
    :: tsv_template=@
    :: cs_sharedtask_dir=@
    :: evaluation_script_dir=@
    :: is_crfsuite=@
    < results=(Model: autoencoder=$auto_test_labels@AutoencoderPosInduction
                      crfsuite=$test_results@TestCRFSuite)
    > eval_results
    > test_tsv
{
  # converts autoencoder output to tsv
  python $cs_sharedtask_dir/crf_labels_to_tsv.py $results $tsv_template $test_tsv $is_crfsuite
  perl $evaluation_script_dir/evaluateOffsets.pl $test_tsv $gold_tsv $eval_results 
}
