#!/usr/bin/perl -w

use strict;
my $script_dir; BEGIN { use Cwd qw/ abs_path /; use File::Basename; $script_dir = dirname(abs_path($0)); push @INC, $script_dir; }
use Getopt::Long;

my $ref="$script_dir/eurogold2.de2en/dev.ref.standard.A";
die unless -f $ref;

`head -150 $ARGV[0] | $script_dir/wpt03-05/conv-pharaoh.pl > tmp.xx.wal`;

system("$script_dir/wpt03-05/wa_eval_align.pl $ref tmp.xx.wal");

unlink 'tmp.xx.wal';

