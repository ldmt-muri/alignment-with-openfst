task wordpair_feats_fwd :: data_file=$data_file dir=$fwd > wordpair_feats_file {
  # use cdec to generate word pair features for a partiuclar test set
  ~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls $data_file
  pushd talign
  make generate-wordpair-features
  gzip -d grammars/wordpairs.f-e.features.gz
  popd
  mv talign "$dir-talign"

  cp "$dir-talign/grammars/wordpairs.f-e.features" wordpair_feats_file
} 

task autoencoder_fwd_align :: test_sents_count=$test_sents_count prefix=$prefix procs=$procs data_file=$data_file dir=$fwd < wordpair_feats_file=$wordpair_feats_file@wordpair_feats_fwd > fwd_align out_err="fwd-out.err" {

  pushd /usr0/home/wammar/alignment-with-openfst/
  # compile
  make -f Makefile-latentCrfAligner
  mv train-latentCrfAligner train-latentCrfAligner-copy
  popd

  # the latent-CRF word alignment model
  mpirun -np $procs /usr0/home/wammar/alignment-with-openfst/train-latentCrfAligner-copy \
  --train-data $data_file \
  --output-prefix "$prefix-fwd" \
  --feat 103 --feat 105 --feat 107 --feat 108 \
  --test-size $test_sents_count \
  --max-em-iter-count 1 \
  --max-lbfgs-iter-count 1 \
  2> $out_err

#  --init-lambda /usr0/home/wammar/alignment-with-openfst/ducttape-files/init-lambda-10 \
#  --init-theta /usr0/home/wammar/alignment-with-openfst/ducttape-files/init-theta-10 \
#  --feat 104 --wordpair-feats $wordpair_feats_file \
#  --minibatch-size 500 \
#  --optimizer adagrad \
#  --max-model1-iter-count 100 \
#  --max-iter-count 1 \
#  --l2-strength 1.0 \
#  --cache-feats false \

  cp "$prefix-fwd.labels" $fwd_align
}

task autoencoder_zbwd_align :: test_sents_count=$test_sents_count prefix=$prefix procs=$procs data_file=$bwd_data_file dir=$bwd > bwd_align out_err="bwd-out.err" {
  pushd /usr0/home/wammar/alignment-with-openfst/
  # compile
  make -f Makefile-latentCrfAligner
  popd

  # use cdec to generate word pair features for a particular test set
  ~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls $data_file
  pushd talign
  make generate-wordpair-features
  gzip -d grammars/wordpairs.f-e.features.gz
  popd
  mv talign "$dir-talign"

  # the latent-CRF word alignment model
  mpirun -np $procs /usr0/home/wammar/alignment-with-openfst/train-latentCrfAligner \
  --train-data $data_file \
  --output-prefix "$prefix-bwd" \
  --feat 102 --feat 103 --feat 104 --feat 105 --feat 106 --feat 107 --feat 108 --feat 109 \
  --wordpair-feats "$dir-talign/grammars/wordpairs.f-e.features" \
  --test-size $test_sents_count \
  --no-direct-dep-btw-hidden-labels \
  2> $out_err

  cp "$prefix-bwd.labels" $bwd_align

  # example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
}

task autoencoder_align :: data_file=$data_file < fwd_align=$fwd_align@autoencoder_fwd_align bwd_align=$bwd_align@autoencoder_zbwd_align > sym_align {
  /usr0/home/wammar/cdec/utils/atools -i $fwd_align -j $bwd_align -c grow-diag-final-and > $sym_align
}

task fast_align :: data_file=$data_file > fwd_align bwd_align sym_align {
  /usr0/home/wammar/cdec/word-aligner/fast_align -i $data_file -d -v -r -o > $bwd_align
  /usr0/home/wammar/cdec/word-aligner/fast_align -i $data_file -d -v -o > $fwd_align
  /usr0/home/wammar/cdec/utils/atools -i $fwd_align -j $bwd_align -c grow-diag-final-and > $sym_align
}

task giza_align :: giza_bin=$giza_bin moses_train_script=$moses_train_script data_file=$data_file src=$src tgt=$tgt procs=$procs > output_dir sym_align fwd_align bwd_align {

  $moses_train_script --first-step 1 --last-step 3 --parallel --root-dir $output_dir --corpus $data_file --f $src --e $tgt --external-bin-dir $giza_bin --mgiza $giza_bin --mgiza-cpus=$procs --alignment srctotgt 
  cp $output_dir/model/aligned.srctotgt $fwd_align

  $moses_train_script --first-step 3 --last-step 3 --parallel --root-dir $output_dir --corpus $data_file --f $src --e $tgt --external-bin-dir $giza_bin --mgiza $giza_bin --mgiza-cpus=$procs --alignment tgttosrc
  cp $output_dir/model/aligned.tgttosrc $bwd_align

  /usr0/home/wammar/cdec/utils/atools -i $fwd_align -j $bwd_align -c grow-diag-final-and > $sym_align

}

task eval_align :: eval_script=$eval_script gold_file=$gold_file < fwd_align=$fwd_align@autoencoder_fwd_align > aer_file="aer" {
  # compute aer
  $eval_script \
    $fwd_align \
    $gold_file \
    > $aer_file
}

task train_cdec :: procs=$procs data_file=$data_file dev_file=$dev_file test_file=$test_file mono_file=$mono_file cdec_ini_file=$cdec_ini_file < sym_align=$sym_align@giza_align > extract_ini_file klm_file="klm" dev_sgm test_sgm lm_file suffix_array dev_grammar test_grammar cdec_ini_ofile {
  export PYTHONPATH=`echo ~/cdec/python/build/lib.*`
  echo "create extract ini file and suffix array from training data $data_file and alignments $sym_align"
  python -m cdec.sa.compile -b $data_file -a $sym_align -c extract_ini_file -o $suffix_array
  echo "create dev set grammar and suffix array"
  python -m cdec.sa.extract -c extract_ini_file -g $dev_grammar -j $procs -z < $dev_file > $dev_sgm
  echo "create test set grammar and suffix array"
  python -m cdec.sa.extract -c extract_ini_file -g $test_grammar -j $procs -z < $test_file > $test_sgm
  /usr0/home/wammar/cdec/klm/lm/builder/builder --order 3 < $mono_file > $lm_file
  /usr0/home/wammar/cdec/klm/lm/build_binary $lm_file $klm_file
  cp $cdec_ini_file $cdec_ini_ofile
  echo "feature_function=KLanguageModel $klm_file" >> $cdec_ini_ofile
}

task tune_mira :: procs=$procs < dev_sgm=$dev_sgm@train_cdec test_sgm=$test_sgm@train_cdec cdec_ini_ofile=$cdec_ini_ofile@train_cdec > mira_out_dir {
  /usr0/home/wammar/cdec/training/mira/mira.py \
    -d $dev_sgm \
    -t $test_sgm \
    -c $cdec_ini_ofile \
    -j $procs \
    -o $mira_out_dir
}

plan plan_giza {
  reach giza_align
}

plan plan_fastalign {
  reach fast_align
}

plan plan_autoencoder {
  reach eval_align
}

plan plan_cdec {
  reach tune_mira
}

global {
  giza_bin=/opt/tools/mgizapp-0.7.2/bin/
  moses_train_script=/usr0/home/wammar/mosesdecoder/scripts/training/train-model.perl
  eval_script=/usr0/home/wammar/alignment-with-openfst/data/czen-manual-alignments/eval-czen.pl
  gold_file=/usr0/home/wammar/alignment-with-openfst/data/czen-manual-alignments/czen.wal
  mono_file=/usr1/home/wammar/monolingual/plain-en/news.2012.en.shuffled
  data_file=/usr0/home/wammar/alignment-with-openfst/data/90k.cz-en
  bwd_data_file=/usr0/home/wammar/alignment-with-openfst/data/90k.en-cz
  dev_file=/usr0/home/wammar/alignment-with-openfst/data/dev.cz-en
  test_file=/usr0/home/wammar/alignment-with-openfst/data/dev2.cz-en
  prefix=90k
  cdec_ini_file=/usr0/home/wammar/alignment-with-openfst/ducttape-files/cdec.ini
  procs=24
  src=cz
  tgt=en
  fwd=cz-en
  bwd=en-cz
  ducttape_structure=flat
  test_sents_count=515
}

