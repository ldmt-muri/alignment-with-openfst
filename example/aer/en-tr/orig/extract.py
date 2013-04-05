#!/usr/bin/env python
import sys

def main():
    for sentence, line in enumerate(sys.stdin):
        points = line.split(';')
        for point in points:
            if not point.strip(): continue
            al = map(int, point.split())
            en = al[0]
            if en == 0: continue # skip NULL
            for tr in al[1:]:
                if tr == 0: continue # skip NULL
                print('{0:04d} {1} {2} S'.format(sentence+1, en, tr))

if __name__ == '__main__':
    main()
