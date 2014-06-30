#!/usr/bin/env ducttape

global {
  ducttape_experimental_submitters=enable
  ducttape_experimental_imports=enable

  # autoencoder hyperparams
  l2_strength=1000 #(L2Strength: zero="" ten=10.0 hundred=100.0 thousand=1000.0 ten_thousand=10000.0 hundred_thousand=100000.0 )
  l1_strength=0 #(L1Strength: zero="" point_o_one=0.01 point_one=0.1 one=1.0 ten=10.0 hundred=100.0 )
  dirichlet_alpha=0.1
  lbfgs_itercount=(LbfgsItercount: one=1 two=2 four=4)
  em_itercount=(EmItercount: one=1 two=2 four=4)
  #coord_itercount=(CoordItercount: normal=50 zero=0 ten=10)
  test_with_crf_only=""
  optimize_lambdas_first=(OptimizeLambdasFirst: yes=true no=false)
  precomputed_features=none
  min_relative_diff=0.000001
  baby_steps=""
  conll_field_reconstructed=2
  wordpair_feats_file=""
    
  # features
  head_child_token_set=(HeadChildTokenSet: yes=1 no="")
  head_child_pos_set=(HeadChildPosSet: yes=1 no="")
  head_child_token=(HeadChildToken: yes=1 no="")
  head_child_pos=(HeadChildPos: yes=1 no="")
  head_pos=(HeadPos: yes=1 no="")
  inbetween_pos=(InbetweenPos: no="" yes=1)
  adjacent_pos=(AdjacentPos: no="" yes=1)
  log_alignment_jump=1
  alignment_jump=""

  # machine:
  ducttape_output="/usr2/home/wammar/crf-auto-parser/ptb/"
  alignment_with_openfst_dir="/usr0/home/wammar/alignment-with-openfst"
  cdec_dir="/usr0/home/wammar/cdec/"
  
  # initial autoencoder params
  init_lambda=""

  # general
  cores=10

  # dataset
  in_conll="/usr1/home/wammar/treebanks/ptb/dmv/devw+trainw10.conll"
  text="/usr0/home/wammar/alignment-with-openfst/parsing/data/english-sample"
  test_sents_count="1700"
  output_prefix="ptb"

}

#import ../submitters.tape

task WordAlign
    :: conll_field_reconstructed=@
    :: in_conll=@
    :: text=@
    :: cdec_dir=@
    :: alignment_with_openfst_dir=@
    > out_conll
    > theta_init
    > eval
#    :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  # first, convert in_conll to text3
  python ~/wammar-utils/convert-conll-to-text.py -i $in_conll -o align_me.txt -c $conll_field_reconstructed
  python ~/wammar-utils/paste.py -d " ||| " -i align_me.txt align_me.txt -o align_me.parallel
  rm align_me.txt
  corpus_length=$(cat align_me.parallel | wc -l)
  # then make the plain text parallel 
  python ~/wammar-utils/paste.py -d " ||| " -i $text $text -o text.parallel
  # then append the plain text corpus
  if [$conll_field_reconstructed -eq 2]; then
    cat text.parallel >> align_me.parallel
  fi
  #rm text.parallel
  # then train with fast align
  ~/cdec/word-aligner/fast_align --no_self_alignments -i align_me.parallel -d -v -o -p $theta_init > align_me.align
  #rm align_me.parallel
  sed -i 's/<eps>/__ROOT__/g' $theta_init
  head -n $corpus_length align_me.align > align_me.align.head
  python ~/wammar-utils/word-alignments-to-dependency-parses.py -w align_me.align.head -ci $in_conll -co $out_conll
  #rm align_me.align align_me.align.head
  perl $alignment_with_openfst_dir/parsing/eval07.pl -q -g $in_conll -s $out_conll > $eval
}

task BuildLatentCrfParser
    :: alignment_with_openfst_dir=@
    > executable
   # :: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=1g .q=shared 
{

  pushd $alignment_with_openfst_dir
  make -f Makefile-latentCrfParser
  cp parsing/train-latentCrfParser $executable
  popd
}

task CrfAutoencoderParse
    > eval
    :: conll_field_reconstructed=@
    :: min_relative_diff=@
    :: alignment_with_openfst_dir=@
    :: test_sents_count=@
    :: procs=$cores
    < in_conll=(InitTheta: yes=$in_conll no=$out_conll@WordAlign)
    :: init_supervised=(InitTheta: yes="" no=1)
    < init_theta=$theta_init@WordAlign
    :: init_lambda=$init_lambda
    :: l2_strength=@
    :: l1_strength=@
    :: dirichlet_alpha=@
    :: test_with_crf_only=@
    :: lbfgs_itercount=@
    :: em_itercount=@
    :: optimize_lambdas_first=@
    :: output_prefix=@
    :: head_child_token=@
    :: head_child_pos=@
    :: head_child_token_set=@
    :: head_child_pos_set=@
    :: head_pos=@
    :: inbetween_pos=@
    :: adjacent_pos=@
    :: log_alignment_jump=@
    :: alignment_jump=@
    < executable=$executable@BuildLatentCrfParser
    > out_conll
    > out_err 
    :: baby_steps=@
    :: wordpair_feats_file=@
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{

  variational=1

  echo init_supervised=$init_supervised

  # the latent-CRF dependency parsing model
  command="mpirun -np $procs $executable --train-data $in_conll --output-prefix $output_prefix 
  --test-size $test_sents_count --max-iter-count 55 --min-iter-count 50 --min-relative-diff $min_relative_diff
  --one-based-conll-field-id-reconstructed $conll_field_reconstructed --feature-gaussian-mean /usr0/home/wammar/alignment-with-openfst/parsing/universal-biases.data --generate-child-and-direction true"
  #--optimizer adagrad --minibatch-size 1000 
  #   

  if [ $baby_steps ]; then
    command="$command --baby-steps true"
  fi

  if [ $wordpair_feats_file ]; then
    command="$command --wordpair-feats $wordpair_feats_file"
  fi

  if [ "$init_supervised" ]; then
    command="$command --supervised true" 
  else
    command="$command --init-theta $init_theta"
  fi

  if [ $init_lambda ]; then
    command="$command --init-lambda $init_lambda"
  fi

  if [ $l2_strength ]; then
    command="$command --l2-strength $l2_strength"
  fi

  if [ $l1_strength ]; then
    command="$command --l1-strength $l1_strength"
  fi

  if [ $dirichlet_alpha ]; then
    command="$command --dirichlet-alpha $dirichlet_alpha"
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

  if [ $lbfgs_itercount ]; then
    command="$command --max-lbfgs-iter-count $lbfgs_itercount"
  fi 

  if [ $optimize_lambdas_first ]; then
    command="$command --optimize-lambdas-first $optimize_lambdas_first"
  fi

  if [ $head_child_token ]; then
    command="$command --feat HC_TOKEN --feat CH_TOKEN"
  fi
  
  if [ $head_child_pos ]; then
    command="$command --feat HC_POS --feat CH_POS"
  fi
  
  if [ $head_child_token_set ]; then
    command="$command --feat HEAD_CHILD_TOKEN_SET"
  fi
  
  if [ $head_child_pos_set ]; then
    command="$command --feat HEAD_CHILD_POS_SET"
  fi
  
  if [ $head_pos ]; then
    command="$command --feat HEAD_POS"
  fi
    
  if [ $inbetween_pos ]; then
    command="$command --feat CXH_POS --feat HXC_POS"
  fi
  
  if [ $adjacent_pos ]; then
    command="$command --feat CXxH_POS --feat HXxC_POS --feat CxXH_POS --feat HxXC_POS --feat XHC_POS --feat XCH_POS --feat HCX_POS --feat CHX_POS"
  fi
  
  if [ $log_alignment_jump ]; then
    command="$command --feat LOG_ALIGNMENT_JUMP --feat POS_PAIR_DISTANCE"
  fi
  
  if [ $alignment_jump ]; then
    command="$command --feat ALIGNMENT_JUMP"
  fi

  echo "executing $command..."
  $command 2> $out_err

  cp $output_prefix.labels $out_conll

  perl $alignment_with_openfst_dir/parsing/eval07.pl -q -g $in_conll -s $out_conll > $eval
}

plan Full {
    reach CrfAutoencoderParse via (HeadChildToken: yes) * (HeadChildPos: yes) * (HeadPos: yes) * (AdjacentPos: yes) * (InbetweenPos: yes) * (OptimizeLambdasFirst: yes) * (HeadChildPosSet: no) * (HeadChildTokenSet: no) * (InitTheta: no) * (LbfgsItercount: four) * (EmItercount: four)
} 

