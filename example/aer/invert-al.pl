#!/usr/bin/perl -w
use strict;
while(<>) {
  chomp;
  my @aps = split /\s+/;
  my @iaps=();
  for my $ap (@aps) {
    my ($a, $b) = split /-/, $ap;
    push @iaps, "$b-$a";
  }
  print "@iaps\n";
}
