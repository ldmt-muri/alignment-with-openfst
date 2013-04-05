#!/usr/bin/perl -w

# $Id: run-test-suite.pl 1853 2008-06-17 17:32:19Z redpony $

use strict;
my $script_dir; BEGIN { use Cwd qw/ abs_path /; use File::Basename; $script_dir = dirname(abs_path($0)); push @INC, $script_dir; }
use Getopt::Long;

my $ref="$script_dir/mt03/mt03.no-waw.al";
die unless -f $ref;

`cat $ref | $script_dir/wpt03-05/conv-pharaoh.pl > tmp.yy.wal`;
`head -200 $ARGV[0] | $script_dir/wpt03-05/conv-pharaoh.pl > tmp.xx.wal`;

system("$script_dir/wpt03-05/wa_eval_align.pl tmp.yy.wal tmp.xx.wal");

unlink 'tmp.yy.wal';
unlink 'tmp.xx.wal';

