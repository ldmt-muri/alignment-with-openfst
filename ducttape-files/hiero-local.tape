#!/usr/bin/env ducttape

global {
  ducttape_experimental_submitters=enable
  ducttape_experimental_imports=enable

  # autoencoder hyperparams
  l2_strength=(L2: zero="" point_o_one=0.01 point_one=0.1 one=1.0 ten=10.0 hundred=100.0 thousand=1000.0 three_thousand=3000.0 ten_thousand=10000.0 thirty_thousand=30000.0 hundred_thousand=100000.0 million=1000000.0)
  dirichlet_alpha=(DirichletAlpha: one=1.0 zero_point_five=0.5 one_point_five=1.5)
  init_learning_rate=(InitLearningRate: one=1.0 point_five=0.5 point_three=0.3 point_o_three=0.03)
  max_lambda_epoch_count=(MaxLambdaEpochCount: one=1 two=2 four=4 eight=8)
  em_itercount=(EmItercount: one=1 zero=0 two=2 four=4 eight=8)
  #coord_itercount=(CoordItercount: normal=50 zero=0 ten=10)
  test_with_crf_only=false
  optimize_lambdas_first=true
  precomputed_features=(PrecomputedFeatures: all=all dyer11=dyer11 none=none)
  use_other_aligners=(UseOtherAligners: no="" yes=yes)
  use_src_bigrams=(UseSrcBigrams: no="" yes=yes)
  sym_heuristic=(SymHeuristic:
                      grow_diag_final="grow-diag-final"
                      grow_diag_final_and="grow-diag-final-and"
                      grow_diag="grow-diag"
                      intersect="intersect"
    alignment_with_openfst_dir="/home/wammar/stochastic-crf-autoencoder/alignment-with-openfst/"
    ch
                      union="union")
  lambda_optimizer=(LambdaOptimizer: sgd="sgd" lbfgs="lbfgs")
}

#import ../submitters.tape

task Tokenize
    < corpus=(DataSection:
                train=$train_corpus
                tune=$tune_corpus
                test=$test_corpus)
    > tokenized_corpus
    :: tokenize_corpus=@
    :: cdec_dir=@
#    :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  if [[ $tokenize_corpus == "false" ]]; then
    cp $corpus $tokenized_corpus
  else
    cat $corpus | $cdec_dir/corpus/tokenize-anything.sh > $tokenized_corpus
  fi
}

task Lowercase
    < corpus=$tokenized_corpus@Tokenize
    > lowercased_corpus
    :: lowercase_corpus=@
    :: cdec_dir=@
#    :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  if [[ $lowercase_corpus == "false" ]]; then
    cp $corpus $lowercased_corpus
  else
    cat $corpus | $cdec_dir/corpus/lowercase.pl > $lowercased_corpus
  fi
}

task FilterCorpus
    < corpus=$lowercased_corpus@Lowercase[DataSection:train]
    > filtered_corpus
    :: max_sentence_length=@
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  if (( $max_sentence_length > 0 )); then
    $cdec_dir/corpus/filter-length.pl -$max_sentence_length $corpus > $filtered_corpus
  else
    cp $corpus $filtered_corpus
  fi
}

task FastAlignS2T
    < corpus=$filtered_corpus@FilterCorpus
    > alignment
    :: cdec_dir=@
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=32g .q=shared 
{
  $cdec_dir/word-aligner/fast_align -i $corpus -d -v -o > $alignment
}

task FastAlignT2S
    < corpus=$filtered_corpus@FilterCorpus
    > alignment
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=32g .q=shared
 {
  $cdec_dir/word-aligner/fast_align -i $corpus -d -v -o -r > $alignment
}

task GizaAlignS2T
    < corpus=$filtered_corpus@FilterCorpus
    > alignment
    > output_dir
    :: giza_bin=@
    :: moses_train_script=@
    :: cores=@
    :: src=@
    :: tgt=@
    :: cdec_dir=@
  #  :: .submitter=torque_normal .walltime="12:00:00" .cpus=1 .vmem=32g .q=normal 
{

  $cdec_dir/corpus/cut-corpus.pl 1 $corpus > $corpus.$src
  $cdec_dir/corpus/cut-corpus.pl 2 $corpus > $corpus.$tgt

  $moses_train_script --first-step 1 --last-step 3 --parallel --root-dir $output_dir --corpus $corpus --f $src --e $tgt --external-bin-dir $giza_bin --mgiza $giza_bin --mgiza-cpus=$cores --alignment srctotgt 
  cp $output_dir/model/aligned.srctotgt $alignment
}

task GizaAlignT2S
    < corpus=$filtered_corpus@FilterCorpus
    < output_dir=$output_dir@GizaAlignS2T
    > alignment
    :: giza_bin=@
    :: moses_train_script=@
    :: cores=@
    :: src=@
    :: tgt=@
    :: cdec_dir=@
  #  :: .submitter=torque_normal .walltime="12:00:00" .cpus=32 .vmem=32g .q=normal
 {

  $cdec_dir/corpus/cut-corpus.pl 1 $corpus > $corpus.$src
  $cdec_dir/corpus/cut-corpus.pl 2 $corpus > $corpus.$tgt

  $moses_train_script --first-step 3 --last-step 3 --parallel --root-dir $output_dir --corpus $corpus --f $src --e $tgt --external-bin-dir $giza_bin --mgiza $giza_bin --mgiza-cpus=$cores --alignment tgttosrc 
  cp $output_dir/model/aligned.tgttosrc $alignment
}

task GenerateWordpairFeatsS2T 
    :: data_file=$train_corpus 
    :: mkcls_bin=@
    :: cdec_dir=@
    :: precomputed_features=@
    :: src_brown_clusters=@
    :: tgt_brown_clusters=@
    :: alignment_with_openfst_dir=@
    > wordpair_feats_file 
    > wordpair_feats_file_cdec 
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=32 .vmem=32g .q=normal 
{
  # use cdec to generate word pair features for a partiuclar test set
  $cdec_dir/word-aligner/aligner.pl --mkcls=$mkcls_bin $data_file
  pushd talign
  if [ $precomputed_features == none ]; then
    cd grammars
    touch wordpairs.f-e.features
  else
    if [ $precomputed_features == all ]; then
      make generate-wordpair-features USE_AFFIXES=1
    else
      make generate-wordpair-features USE_AFFIXES=0
    fi
    gzip -d grammars/wordpairs.f-e.features.gz
  fi
  popd
  
  cp "talign/grammars/wordpairs.f-e.features" $wordpair_feats_file_cdec

  python $alignment_with_openfst_dir/core/augment_wordpair_feats.py -i $wordpair_feats_file_cdec -sb $src_brown_clusters -tb $tgt_brown_clusters -o $wordpair_feats_file
  
} 

task GenerateWordpairFeatsT2S 
    :: data_file=$train_corpus 
    :: mkcls_bin=@
    :: wammar_utils_dir=@
    :: tgt=@
    :: src=@
    :: precomputed_features=@
    :: cdec_dir=@
    :: src_brown_clusters=$tgt_brown_clusters
    :: tgt_brown_clusters=$src_brown_clusters
    :: alignment_with_openfst_dir=@
    > wordpair_feats_file 
    > wordpair_feats_file_cdec 
    #:: .submitter=torque_normal .walltime="12:00:00" .cpus=32 .vmem=64g .q=normal 
{  
  python $wammar_utils_dir/horizontal-split-parallel-corpus.py -i $data_file -o src tgt
  python $wammar_utils_dir/paste.py -i tgt src -d " ||| " -o reverse.$tgt-$src

  # use cdec to generate word pair features for a partiuclar test set
  $cdec_dir/word-aligner/aligner.pl --mkcls=$mkcls_bin reverse.$tgt-$src
  pushd talign
  if [ $precomputed_features == none ]; then
    cd grammars
    touch wordpairs.f-e.features
  else
    if [ $precomputed_features == all ]; then
      make generate-wordpair-features USE_AFFIXES=1
    else
      make generate-wordpair-features USE_AFFIXES=0
    fi
    gzip -d grammars/wordpairs.f-e.features.gz
  fi
  popd
  
  cp "talign/grammars/wordpairs.f-e.features" $wordpair_feats_file_cdec

  python $alignment_with_openfst_dir/core/augment_wordpair_feats.py -i $wordpair_feats_file_cdec -sb $src_brown_clusters -tb $tgt_brown_clusters -o $wordpair_feats_file
  
} 

task BuildLatentCrfAligner
    :: alignment_with_openfst_dir=@
    > executable
   # :: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=1g .q=shared 
{

  pushd $alignment_with_openfst_dir
  make -f Makefile-latentCrfAligner
  cp alignment/train-latentCrfAligner $executable
  popd
}

task AutoencoderAlignS2T
    :: alignment_with_openfst_dir=@
    :: test_sents_count=@
    :: procs=$cores
    :: data_file=$train_corpus
    :: init_theta=$init_theta_fwd
    :: init_lambda=$init_lambda_fwd
    :: l2_strength=@
    :: dirichlet_alpha=@
    :: init_learning_rate=@
    :: test_with_crf_only=@
    :: max_lambda_epoch_count=@
    :: em_itercount=@
    :: optimize_lambdas_first=@
    :: fwd_giza_alignments=@
    :: bwd_giza_alignments=@
    :: sym_giza_alignments=@
    :: fwd_fast_alignments=@
    :: bwd_fast_alignments=@
    :: sym_fast_alignments=@
    :: use_other_aligners=@
    :: use_src_bigrams=@
    :: tgt_brown_clusters=@
    :: output_prefix=@
    :: lambda_optimizer=@
    < executable=$executable@BuildLatentCrfAligner
    < wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeatsS2T 
    > alignment 
    > out_err 
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{

  model1_itercount="5"
  variational=""

  # the latent-CRF word alignment mode
  command="mpirun -np $procs $executable
  --train-data $data_file 
  --output-prefix $output_prefix 
  --feat PRECOMPUTED --feat DIAGONAL_DEVIATION --feat LOG_ALIGNMENT_JUMP
  --test-size $test_sents_count
  --max-model1-iter-count $model1_itercount
  --lambda-optimizer $lambda_optimizer
  --max-iter-count 10"

  # --feat SRC0_TGT0
  #--tgt-word-classes-filename $tgt_brown_clusters
  # hiding --feat SRC0_TGT0 
  # --feat SYNC_START --feat SYNC_END  
  #  --minibatch-size 1000

  if [ $wordpair_feats_file ]; then
    command="$command --wordpair-feats $wordpair_feats_file"
  fi

  
  if [ $init_theta ]; then
    command="$command --init-theta $init_theta"  
  fi

  if [ $init_lambda ]; then
    command="$command --init-lambda $init_lambda"
  fi

  if [ $l2_strength ]; then
    command="$command --l2-strength $l2_strength"
  fi

  if [ $dirichlet_alpha ]; then
    command="$command --dirichlet-alpha $dirichlet_alpha"
  fi

  if [ $init_learning_rate ]; then
    command="$command --lambda-learning-rate $init_learning_rate"
  fi

  if [ $variational ]; then
    command="$command --variational-inference $variational"
  fi

  if [ $test_with_crf_only ]; then
    command="$command --test-with-crf-only $test_with_crf_only"
  fi

  if [ $em_itercount ]; then
    command="$command --max-em-iter-count $em_itercount"
  fi

  if [ $max_lambda_epoch_count ]; then
    command="$command --max-lambda-updates-epoch-count $max_lambda_epoch_count"
  fi 

  if [ $optimize_lambdas_first ]; then
    command="$command --optimize-lambdas-first $optimize_lambdas_first"
  fi

  if [ $fwd_giza_alignments ]; then
    command="$command --other-aligners-output-filenames $fwd_giza_alignments"
  fi

  if [ $bwd_giza_alignments ]; then
    command="$command --other-aligners-output-filenames $bwd_giza_alignments"
  fi

  if [ $sym_giza_alignments ]; then
    command="$command --other-aligners-output-filenames $sym_giza_alignments"
  fi

  if [ $fwd_fast_alignments ]; then
    command="$command --other-aligners-output-filenames $fwd_fast_alignments"
  fi

  if [ $bwd_fast_alignments ]; then
    command="$command --other-aligners-output-filenames $bwd_fast_alignments"
  fi

  if [ $sym_fast_alignments ]; then
    command="$command --other-aligners-output-filenames $sym_fast_alignments"
  fi

  if [ $use_other_aligners ]; then
    command="$command --feat OTHER_ALIGNERS"
  fi

  if [ $use_src_bigrams ]; then
    command="$command --feat SRC_BIGRAM"
  fi

  echo "executing $command..."
  $command 2> $out_err


  cp $output_prefix.labels $alignment

}

task AutoencoderAlignT2S
    :: output_prefix=@
    :: test_sents_count=@ 
    :: procs=$cores
    :: data_file=$train_corpus
    :: init_theta=$init_theta_bwd
    :: init_lambda=$init_lambda_bwd
    :: l2_strength=@
    :: dirichlet_alpha=@
    :: init_learning_rate=@
    :: test_with_crf_only=@
    :: max_lambda_epoch_count=@
    :: em_itercount=@
    :: optimize_lambdas_first=@
    :: fwd_giza_alignments=@
    :: bwd_giza_alignments=@
    :: sym_giza_alignments=@
    :: fwd_fast_alignments=@
    :: bwd_fast_alignments=@
    :: sym_fast_alignments=@
    :: use_other_aligners=@
    :: use_src_bigrams=@
    :: src_brown_clusters=@
    :: lambda_optimizer=@
    < wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeatsT2S 
    < executable=$executable@BuildLatentCrfAligner
    > alignment 
    > out_err 
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{

  model1_itercount="5"
  variational="true"

  # the latent-CRF word alignment mode
  command="mpirun -np $procs $executable 
  --train-data $data_file 
  --output-prefix $output_prefix 
  --feat PRECOMPUTED --feat DIAGONAL_DEVIATION --feat LOG_ALIGNMENT_JUMP
  --test-size $test_sents_count 
  --reverse true
  --max-model1-iter-count $model1_itercount
  --lambda-optimizer $lambda_optimizer 
  --max-iter-count 10
  --use-averaged-gradient false"
  #--tgt-word-classes-filename $src_brown_clusters
  #   --feat SRC0_TGT0
  #   --feat SYNC_START --feat SYNC_END
  #--minibatch-size 1000
 
  if [ $wordpair_feats_file ]; then
    command="$command --wordpair-feats $wordpair_feats_file"
  fi
  
  if [ $init_theta ]; then
    command="$command --init-theta $init_theta"
  fi

  if [ $init_lambda ]; then
    command="$command --init-lambda $init_lambda"
  fi 

  if [ $l2_strength ]; then
    command="$command --l2-strength $l2_strength"
  fi

  if [ $dirichlet_alpha ]; then
    command="$command --dirichlet-alpha $dirichlet_alpha"
  fi

  if [ $init_learning_rate ]; then
    command="$command --lambda-learning-rate $init_learning_rate"
  fi

  if [ $variational ]; then
    command="$command --variational-inference $variational"
  fi

  if [ $test_with_crf_only ]; then
    command="$command --test-with-crf-only $test_with_crf_only"
  fi
 
  if [ $em_itercount ]; then
    command="$command --max-em-iter-count $em_itercount"
  fi

  if [ $max_lambda_epoch_count ]; then
    command="$command --max-lambda-updates-epoch-count $max_lambda_epoch_count"
  fi 

  if [ $optimize_lambdas_first ]; then
    command="$command --optimize-lambdas-first $optimize_lambdas_first"
  fi

  if [ $fwd_giza_alignments ]; then
    command="$command --other-aligners-output-filenames $fwd_giza_alignments"
  fi

  if [ $bwd_giza_alignments ]; then
    command="$command --other-aligners-output-filenames $bwd_giza_alignments"
  fi

  if [ $sym_giza_alignments ]; then
    command="$command --other-aligners-output-filenames $sym_giza_alignments"
  fi

  if [ $fwd_fast_alignments ]; then
    command="$command --other-aligners-output-filenames $fwd_fast_alignments"
  fi

  if [ $bwd_fast_alignments ]; then
    command="$command --other-aligners-output-filenames $bwd_fast_alignments"
  fi

  if [ $sym_fast_alignments ]; then
    command="$command --other-aligners-output-filenames $sym_fast_alignments"
  fi

  if [ $use_other_aligners ]; then
    command="$command --feat OTHER_ALIGNERS"
  fi

  if [ $use_src_bigrams ]; then
    command="$command --feat SRC_BIGRAM"
  fi

  echo "executing $command..."
  $command  2> $out_err

  cp $output_prefix.labels $alignment
}

task Symmetrize
    < s2tAlignment=(Aligner:
                      giza=$alignment@GizaAlignS2T
                      fast=$alignment@FastAlignS2T
                      autoencoder=$alignment@AutoencoderAlignS2T
                      giza_none=$alignment@GizaAlignS2T
                      none_giza=$alignment@GizaAlignT2S
                      fast_none=$alignment@FastAlignS2T
                      none_fast=$alignment@FastAlignT2S
                      autoencoder_none=$alignment@AutoencoderAlignS2T
                      none_autoencoder=$alignment@AutoencoderAlignT2S
                      autoencoder_giza=$alignment@AutoencoderAlignS2T
                      autoencoder_fast=$alignment@AutoencoderAlignS2T
                      fast_autoencoder=$alignment@FastAlignS2T
                      giza_autoencoder=$alignment@GizaAlignS2T)
    < t2sAlignment=(Aligner:
                      giza=$alignment@GizaAlignT2S
                      fast=$alignment@FastAlignT2S
                      autoencoder=$alignment@AutoencoderAlignT2S
                      giza_none=$alignment@GizaAlignS2T
                      none_giza=$alignment@GizaAlignT2S
                      fast_none=$alignment@FastAlignS2T
                      none_fast=$alignment@FastAlignT2S
                      autoencoder_none=$alignment@AutoencoderAlignS2T
                      none_autoencoder=$alignment@AutoencoderAlignT2S
                      autoencoder_giza=$alignment@GizaAlignT2S
                      autoencoder_fast=$alignment@FastAlignT2S
                      fast_autoencoder=$alignment@AutoencoderAlignT2S
                      giza_autoencoder=$alignment@AutoencoderAlignT2S)
    > alignment
    :: sym_heuristic=@
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  $cdec_dir/utils/atools -i $s2tAlignment -j $t2sAlignment -c $sym_heuristic > $alignment
}

task BuildSuffixArray
     < corpus=$filtered_corpus@FilterCorpus
     < alignment=$alignment@Symmetrize  
     > ini
     > suffix_array
    :: cdec_dir=@
#    :: .submitter=torque_shared .walltime="48:00:00" .cpus=1 .vmem=33g .q=shared 
{
  export PYTHONPATH=`echo $cdec_dir/python/build/lib.*`
  python -m cdec.sa.compile -b $corpus -a $alignment -c $ini -o $suffix_array
}

task ExtractGrammars
    < corpus=(ExtractSection:
                tune=$lowercased_corpus@Lowercase[DataSection:tune]
                test=$lowercased_corpus@Lowercase[DataSection:test])
    < ini=@BuildSuffixArray
    > wrapped_corpus
    > grammar_dir
    :: cores=@
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="24:00:00" .cpus=1 .vmem=33g .q=shared 
{
  export PYTHONPATH=`echo $cdec_dir/python/build/lib.*`
  cat $corpus | python -m cdec.sa.extract -c $ini -g $grammar_dir -j $cores -z > $wrapped_corpus
}

task BuildLanguageModel
    < corpus=$filtered_corpus@FilterCorpus
    < lm_data=@
    > language_model
    :: lm_order=@
    :: cdec_dir=@
    :: kenlm_dir=@
  #  :: .submitter=torque_shared .walltime="06:00:00" .cpus=1 .vmem=33g .q=shared 
{
  corpus_target="./corpus_target"
  $cdec_dir/corpus/cut-corpus.pl 2 $corpus > $corpus_target
  cat $corpus_target $lm_data |   $kenlm_dir/bin/lmplz -o $lm_order --interpolate_unigrams --memory=10% > $language_model
}

task CompileLanguageModel
    < arpa=$language_model@BuildLanguageModel
    > language_model
    :: cdec_dir=@
 #   :: .submitter=torque_shared .walltime="06:00:00" .cpus=1 .vmem=33g .q=shared 
{
  $cdec_dir/klm/lm/build_binary $arpa $language_model
}

task MakeCdecIni
    < language_model=(UseCustomLM: no=$language_model@CompileLanguageModel yes=$language_model)
    > cdec_ini {
  echo "formalism=scfg" > $cdec_ini
  echo "add_pass_through_rules=true" >> $cdec_ini
  echo "feature_function=WordPenalty" >> $cdec_ini
  echo "feature_function=KLanguageModel $language_model" >> $cdec_ini
}

task Tune
    < tune_set=$wrapped_corpus@ExtractGrammars[ExtractSection:tune]
    < cdec_ini=@MakeCdecIni
    > mira_work
    > optimized_weights
    :: tune_corpus=@
    :: cores=@
    :: cdec_dir=@
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=60g .q=normal 
{


  ref_count=$(head -n 1 $tune_set | grep -o '|||' | wc -l)
  #cut -f 1 -d '|' $tune_set > ./tune_source
  for i in `seq 1 $ref_count`; do
     cut -f $(expr 3 \* $i + 1) -d '|' $tune_set > ./tune_refs.$i
  done

  initial_weights="./initial_weights"
  echo "SampleCountF 0.0"      >> $initial_weights 
  echo "CountEF 0.0"           >> $initial_weights
  echo "MaxLexEgivenF 0.0"     >> $initial_weights
  echo "EgivenFCoherent 0.0"   >> $initial_weights
  echo "MaxLexFgivenE 0.0"     >> $initial_weights
  echo "PassThrough 0.0"       >> $initial_weights
  echo "PassThrough_1 0.0"     >> $initial_weights
  echo "PassThrough_2 0.0"     >> $initial_weights
  echo "PassThrough_3 0.0"     >> $initial_weights
  echo "PassThrough_4 0.0"     >> $initial_weights
  echo "PassThrough_5 0.0"     >> $initial_weights
  echo "PassThrough_6 0.0"     >> $initial_weights
  echo "IsSingletonF 0.0"      >> $initial_weights
  echo "IsSingletonFE 0.0"     >> $initial_weights
  echo "LanguageModel 0.0"     >> $initial_weights
  echo "LanguageModel_OOV 0.0" >> $initial_weights
  echo "Glue 0.0"              >> $initial_weights
  echo "WordPenalty 0.0"       >> $initial_weights

  $cdec_dir/training/mira/mira.py -j $cores --update-size 250 -o $mira_work --weights ./initial_weights \
    --devset $tune_set --config $cdec_ini

  index=$(ls $mira_work | grep '^weights\.[0-9]*$' | sed 's/^weights\.//' | sort -n | tail -n 1)
  ln -s $mira_work/weights.$index $optimized_weights
}

task Decode
    < test_set=(TuneOrTest:
        test=$wrapped_corpus@ExtractGrammars[ExtractSection:test]
        tune=$wrapped_corpus@ExtractGrammars[ExtractSection:tune])
    < cdec_ini=@MakeCdecIni
    < weights=$optimized_weights@Tune
    > kbest
    > output
    :: k=1000
    :: cores=@
    :: cdec_dir=@
 #   :: .submitter=torque_normal .walltime="6:00:00" .cpus=32 .vmem=60g .q=normal 
{

  cat $test_set | $cdec_dir/corpus/cut-corpus.pl 1 | $cdec_dir/decoder/cdec -c $cdec_ini -w $weights -k $k -r > $kbest
  sed 's/ ||| /\t/g' $kbest | sort -u -k 1,1 -n | cut -f 2 > $output
}

task Evaluate 
    < output=$output@Decode
    < refs=(TuneOrTest: test=$test_corpus tune=$tune_corpus)
    > bleu meteor ter length
    :: multeval=@
    :: meteor_lang=@
   # :: .submitter=torque_shared .walltime="00:20:00" .cpus=1 .vmem=4g .q=shared 
{
  METEORTASK=li
  METEORLANG=$meteor_lang
  scoreFile=scores.txt

  num_refs=$(head -n 1 $refs | grep -o '|||' | wc -l)
  for i in `seq 1 $num_refs`; do
    cut -f $(expr 3 \* $i + 1) -d '|' $refs | sed 's/^\s*//' | sed 's/\s*$//' > refs.$i
  done

  ln -s $(dirname $multeval)/constants .
  echo "executing: $multeval eval --refs refs.* --hyps-baseline $output --meteor.task $METEORTASK --meteor.language $METEORLANG &> $scoreFile"
  $multeval eval --refs refs.* --hyps-baseline $output --meteor.task $METEORTASK --meteor.language $METEORLANG &> $scoreFile
  echo "done executing that command."
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 2 > $bleu
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 3 > $meteor
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 4 > $ter
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 5 > $length
}

task EvaluateAlignment
    :: conv_pharaoh_script=@
    :: aer_eval_script=@
    :: gold_alignment=@
    :: test_sents_count=@ 
    < alignment=$alignment@Symmetrize 
    > aer 
#    :: .submitter=torque_shared .walltime="00:20:00" .cpus=1 .vmem=4g .q=shared 
  {
  head -n $test_sents_count $alignment | $conv_pharaoh_script > temp
  $aer_eval_script    $gold_alignment temp     > $aer
}

#summary EvaluationSummary {
#  of Evaluate > Score {
#    cp $bleu $Score
#  }
#}
