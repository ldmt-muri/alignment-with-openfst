import re
import time
import io
import sys
import argparse
from collections import defaultdict
import fnmatch
import os

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-r", "--root")
argParser.add_argument("-p", "--params_pattern")
args = argParser.parse_args()

#unreg_ll_regex_pattern = re.compile(r'^actually adding an l2 term\.\. before: (.+)')
unreg_ll_regex_pattern = re.compile(r'actually adding an l2 term\.\. before: ([\d.]+)')
reg_ll_regex_pattern = re.compile(r'\.\.after: ([\d.]+)')
m21_regex_pattern = re.compile(r'many-to-one = ([\d.]+)')
vi_regex_pattern = re.compile(r'variation of information = ([\d.]+)')

best_unreg_ll, best_unreg_ll_filename, best_unreg_ll_m21, best_unreg_ll_vi = float('inf'), 'nofile', 0.0, 0.0
best_reg_ll, best_reg_ll_filename, best_reg_ll_m21, best_reg_ll_vi = float('inf'), 'nofile', 0.0, 0.0
best_m21, best_m21_filename, best_m21_vi = 0.0, 'nofile', 0.0

for root, dirs, files in os.walk(args.root):
  for filename in fnmatch.filter(files, 'out_err'):
    if not fnmatch.fnmatch(os.path.join(root, filename), args.params_pattern): continue
    print 'considering ', os.path.join(root, filename)

    # get ll and evaluation metrics of converged model
    unreg_ll, reg_ll, vi, m21 = 0.0, 0.0, 0.0, 0.0
    for match in re.finditer(unreg_ll_regex_pattern, io.open( os.path.join(root, filename) ).read()):
      unreg_ll = float(match.group(1))
    for match in re.finditer(reg_ll_regex_pattern, io.open( os.path.join(root, filename) ).read()):
      reg_ll = float(match.group(1))
    for match in re.finditer(vi_regex_pattern, io.open( os.path.join(root, filename) ).read()):
      vi = float(match.group(1))
    for match in re.finditer(m21_regex_pattern, io.open( os.path.join(root, filename) ).read()):
      m21 = float(match.group(1))
    print 'reg_ll=', reg_ll, ', unreg_ll=', unreg_ll, ', vi=', vi, ', m21=', m21

    # if this is the best unreg ll achieved, update it
    if unreg_ll < best_unreg_ll: 
      best_unreg_ll, best_unreg_ll_filename, best_unreg_ll_m21, best_unreg_ll_vi = unreg_ll, os.path.join(root, filename), m21, vi
      print 'new unreg_ll record!'

    # if this is the best reg ll achileved, update it
    if reg_ll < best_reg_ll:
      best_reg_ll, best_reg_ll_filename, best_reg_ll_m21, best_reg_ll_vi = reg_ll, os.path.join(root, filename), m21, vi
      print 'new reg_ll record!'

    if m21 > best_m21:
      best_m21, best_m21_filename, best_m21_vi = m21, os.path.join(root, filename), vi
      print 'new m21 record!'
    
    print

print 'and the winners are:'
print '--------------------'
print 'best unreg_ll = ', best_unreg_ll
print '  filename    = ', best_unreg_ll_filename
print '  m21         = ', best_unreg_ll_m21
print '  vi          = ', best_unreg_ll_vi
print
print 'best reg_ll = ', best_reg_ll
print '  filename  = ', best_reg_ll_filename
print '  m21       = ', best_reg_ll_m21
print '  vi        = ', best_reg_ll_vi
print
print 'best m21    = ', best_m21
print '  filename  = ', best_m21_filename
print '  vi        = ', best_m21_vi
print '----------------------'
