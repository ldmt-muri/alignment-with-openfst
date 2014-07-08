#!/bin/bash
# 
# File:   remote-build.bash
# Author: wammar
#
# Created on Jun 18, 2013, 11:51:01 PM
#
export MODULEPATH=/opt/modulefiles
source ~/.profile
make -f Makefile-latentCrfPosTagger
