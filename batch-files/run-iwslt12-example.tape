#!/usr/bin/env ducttape

package latentCrf :: .versioner=git .repo=/usr0/home/wammar/alignment-with-openfst .ref=HEAD {
  make
}

task trainer : latentCrf < observed=../example/iwslt12-pos-100.eng > log=../../example/iwslt12.latentCrf.out.log :: prefix=../example/iwslt12.latentCrf.out {
  $latentCrf/train-latentCrfModel $observed $prefix 2> $log
}

# The ducttape_structure global is a special directive to ducttape (ignore it for now)
global {
  ducttape_structure=flat
  ducttape_experimental_packages=true
}

# q:
