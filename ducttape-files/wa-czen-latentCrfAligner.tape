
task autoencoder_fwd_align :: test_sents_count=$test_sents_count prefix=$prefix procs=$procs data_file=$data_file dir=$fwd > fwd_align out_err="fwd-out.err" {
  pushd /usr0/home/wammar/alignment-with-openfst/
  # compile
  make -f Makefile-latentCrfAligner
  popd

  # use cdec to generate word pair features for a partiu7clar test set
  ~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls $data_file
  pushd talign
  make generate-wordpair-features
  gzip -d grammars/wordpairs.f-e.features.gz
  popd
  echo "$dir-talign"
  mv talign "$dir-talign"

  # the latent-CRF word alignment model
  echo "$dir-talign/grammars/wordpairs.f-e.features"
  echo "$prefix-fwd"
  mpirun -np $procs /usr0/home/wammar/alignment-with-openfst/train-latentCrfAligner \
    $data_file \
    none none \
    "$dir-talign/grammars/wordpairs.f-e.features" \
    "$prefix-fwd" \
    $test_sents_count \
    2> $out_err

  echo "$prefix-fwd.labels"
  cp "$prefix-fwd.labels" $fwd_align

  # example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
}

task autoencoder_bwd_align :: test_sents_count=$test_sents_count prefix=$prefix procs=$procs data_file=$bwd_data_file dir=$bwd > bwd_align out_err="bwd-out.err" {
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
    $data_file \
    none none \
    "$dir-talign/grammars/wordpairs.f-e.features" \
    "$prefix-bwd" \
    $test_sents_count \
    2> $out_err

  cp "$prefix-bwd.labels" $bwd_align

  # example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
}

task autoencoder_align :: data_file=$data_file < fwd_align=$fwd_align@autoencoder_fwd_align bwd_align=$bwd_align@autoencoder_bwd_align > sym_align {
  /usr0/home/wammar/cdec/utils/atools -i $fwd_align -j $bwd_align -c grow-diag-final-and > $sym_align
}

task fast_align :: data_file=$data_file > fwd_align bwd_align sym_align {
  /usr0/home/wammar/cdec/word-aligner/fast_align -i $data_file -d -v -r -o > $bwd_align
  /usr0/home/wammar/cdec/word-aligner/fast_align -i $data_file -d -v -o > $fwd_align
  /usr0/home/wammar/cdec/utils/atools -i $fwd_align -j $bwd_align -c grow-diag-final-and > $sym_align
}

task eval_align :: eval_script=$eval_script gold_file=$gold_file < fwd_align=$fwd_align@fast_align > aer_file="aer" {
  # compute aer
  $eval_script \
    $fwd_align \
    $gold_file \
    > $aer_file  
}

task train_cdec :: procs=$procs data_file=$data_file dev_file=$dev_file test_file=$test_file mono_file=$mono_file cdec_ini_file=$cdec_ini_file < sym_align=$sym_align@autoencoder_align > extract_ini_file klm_file="klm" dev_sgm test_sgm lm_file suffix_array dev_grammar test_grammar cdec_ini_ofile {
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

global {
  eval_script=/usr0/home/wammar/alignment-with-openfst/data/czen-manual-alignments/eval-czen.pl
  gold_file=/usr0/home/wammar/alignment-with-openfst/data/czen-manual-alignments/czen.wal
  mono_file=/usr0/home/wammar/alignment-with-openfst/data/10k.cz-en.en
  data_file=/usr0/home/wammar/alignment-with-openfst/data/10k.cz-en
  bwd_data_file=/usr0/home/wammar/alignment-with-openfst/data/10k.en-cz
  dev_file=/usr0/home/wammar/alignment-with-openfst/data/1k.cz-en
  test_file=/usr0/home/wammar/alignment-with-openfst/data/1k.cz-en
  prefix=10k
  cdec_ini_file=/usr0/home/wammar/alignment-with-openfst/ducttape-files/cdec.ini
  procs=32
  src=cz
  tgt=en
  fwd=cz-en
  bwd=en-cz
  ducttape_structure=flat
  test_sents_count=515
}

