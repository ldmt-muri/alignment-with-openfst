#!/usr/bin/perl -w
use strict;


my $sn = 0;
while(<>) {
  chomp;
  if (/^NULL/) {
    $sn++;
    # NULL ({ }) oh ({ 1 3 }) , ({ 2 }) oh ({ }) ! ({ 4 })
    my @blocks = split /\s+\(\{\s+/; shift @blocks;
    shift @blocks;
    my $fsn = sprintf '%04d', $sn;
    my $bn = -1;
    my $first = 1;
    for my $block (@blocks) {
      $block =~ s/^(.*)\s*\}\).*$/$1/;
      $bn++;
      next if $block =~ /^$/;
      my @aps = split /\s+/, $block;
      for my $ap (@aps) { my $i=$ap-1; print "$fsn $bn $i\n"; }
    }
  }
}
