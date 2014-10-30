#!/usr/bin/env bash

export wammar_utils_dir="/usr0/home/wammar/wammar-utils/"
export labeled_test_text="/usr1/home/wammar/pos-data/conll2007/autoencoder-format/english-ptb-2007.tok.4k"
export labeled_test_labels="/usr1/home/wammar/pos-data/conll2007/autoencoder-format/english-ptb-2007.pos.4k"
export labels_dir="/usr3/home/wammar/crf-auto-pos/english4k-final/AutoencoderPosInduction/CoordIterCount.thousand+DirichletAlpha.one+FeatureSet.hk+L2Strength.one+Prefix.other/"
for out_labels in "$labels_dir"/*labels* ; do        
    echo "=========================================="
    echo "evaluating $out_labels"
    echo "=========================================="
    if [ -e "$out_labels" ] ; then 
        # convert tokens and gold labels to token/label format
        python $wammar_utils_dir/combine-token-label-in-one-file.py $labeled_test_text $labeled_test_labels gold_file
        # convert tokens and predicted labels to token/label format
        python $wammar_utils_dir/combine-token-label-in-one-file.py $labeled_test_text "$out_labels" out_file
        python $wammar_utils_dir/score-classes.py gold_file out_file
        python $wammar_utils_dir/score-vm.py gold_file out_file 
    fi
done
