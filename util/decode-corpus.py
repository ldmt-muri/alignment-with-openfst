import re
import time
import io
import sys
from collections import defaultdict

# splits tokens on a whitespace, and outputs unique tokens

vocabFile = io.open(sys.argv[1], encoding='utf8', mode='r')
print 'vocabFile is {0}'.format(sys.argv[1])
intCorpus = io.open(sys.argv[2], encoding='utf8', mode='r')
print 'intCorpus is {0}'.format(sys.argv[2])
textCorpus = io.open(sys.argv[3], encoding='utf8', mode='w')
print 'textCorpus is {0}'.format(sys.argv[3])

# read vocab file into a list
vocab = ['']
counter = 1
for line in vocabFile:
  (nextId, token) = line.strip().split()
  assert counter == int(nextId), 'vocab file has been modified at {0}'.format(nextId)
  counter += 1
  vocab.append(token)
vocabFile.close()

# restore text
lineNumber = 0
for line in intCorpus:
  temp = []
  lineNumber += 1
  for token in line.strip().split():
    intToken = int(token)
    assert intToken >= 1, 'integer-encoded tokens must be positive. violated in line number {0}'.format(lineNumber)
    assert intToken < len(vocab), 'integer-encoded tokens must be less than vocab size. violated in line number {0}'.format(lineNumber)
    temp.append(vocab[intToken])
  textCorpus.write(u'{0}\n'.format(' '.join(temp)))

textCorpus.close()
intCorpus.close()
