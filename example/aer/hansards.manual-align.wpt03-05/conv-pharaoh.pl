#!/usr/bin/perl -w
use strict;


my $sn = 0;
while(<>) {
  chomp;
  $sn++;
  my @aps = split /\s+/;
  for my $ap (@aps) {
    my ($f,$e) = split /\-/, $ap;
    $f++; $e++;
    my $fs = sprintf '%04d', $sn;
    print "$fs $e $f\n";
  }
}
