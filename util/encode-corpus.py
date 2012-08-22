import re
import time
import io
import sys
from collections import defaultdict

# splits tokens on a whitespace, and outputs unique tokens

textCorpus = io.open(sys.argv[1], encoding='utf8', mode='r')
vocabFile = io.open(sys.argv[2], encoding='utf8', mode='w')
intCorpus = io.open(sys.argv[3], encoding='utf8', mode='w')

nextId = 2
vocab = defaultdict(int)
for line in textCorpus:
  temp = []
  tokens = line.strip().split()
  for token in tokens:
    if token not in vocab.keys():
      vocab[token] = nextId
      vocabFile.write(u'{0} {1}\n'.format(nextId, token))
      nextId += 1
    temp.append(str(vocab[token]))
  intCorpus.write(u'{0}\n'.format(' '.join(temp)))
                  
vocabFile.close()
textCorpus.close()
intCorpus.close()
