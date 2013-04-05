#!/usr/bin/env python
import sys
import re

plus_re = re.compile('\+([^\W]|\')+ ', re.UNICODE)
def main():
    for line in sys.stdin:
        sys.stdout.write(plus_re.sub('', line.decode('utf8')).encode('utf8'))

if __name__ == '__main__':
    main()
