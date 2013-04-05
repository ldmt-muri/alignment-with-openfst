#!/usr/bin/perl -w
while(<>) {
 chomp;
 my @aps = split /\s+/;
 shift @aps;
 my @oaps;
 for my $ap (@aps) {
  my $prob = 0;
  if ($ap =~ /^P/) {
    $ap =~ s/^P//;
    $prob = 1;
  }
  my ($a, $b) =split /-/, $ap;
  $a--;
  $b--;
  next if $a < 0 || $b < 0;
  my $x = "$a-$b";
  if ($prob) { $x = "$a?$b"; }
  push @oaps, $x;
 }
 print "@oaps\n";
}
