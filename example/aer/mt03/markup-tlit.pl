#!/usr/bin/perl -w
use strict;
use utf8;
die "Usage: $0 dict.txt < fr-en.al\n" unless scalar @ARGV == 1;
my $dictf = shift @ARGV;
print STDERR "Reading transliteration dictionary from: $dictf\n";
open D, "<$dictf" or die "Can't read $dictf: $!";
binmode(D,":utf8");
my %dict=();
my %ftc;
my %ttc;
while(<D>) {
  chomp;
  my ($src, $flds) = split /\t/;
  $src=~s/^[^ ]+ \|\|\| //;
  my (@trgs) = split / \|\|\| /, $flds;
  my @ot;
  for (my $i = 0; $i < scalar @trgs; $i += 2) {
    push @ot, $trgs[$i];
  }
  for my $trg (@ot) {
    $dict{$src}->{$trg} = 1;
    $ttc{"$src $trg"}++;
  }
}
close D;
print STDERR "Reading aligned reference corpus from STDIN\n";
binmode(STDIN,":utf8");
my $taps = 0;
my $txlit = 0;
my $txlat = 0;
my %tc = ();
while(<>) {
  #Ezp AbrAhym ystqbl ms&wlA AqtSAdyA sEwdyA fy bgdAd . ||| izzet ibrahim meets saudi trade official in baghdad ||| 0-0 1-1 2-2 3-5 4-4 5-3 6-6 7-7
  chomp;
  my ($sf, $se, $sal) = split / \|\|\| /;
  die unless defined $sf;
  die unless defined $se;
  die unless defined $sal;
  my @fs = split /\s+/, $sf;
  my @es = split /\s+/, $se;
  my @aps = split /\s+/, $sal;
  my $first = 1;
  print "$sf ||| $se |||";
  for my $ap (@aps) {
    print ' ';
    if ($ap =~ /^\?/) {
      print $ap;
      next;
    }
    my ($a,$b) =split /-/, $ap;
    my $e = $es[$b];
    my $f = $fs[$a];
    die unless defined $e;
    die unless defined $f;
    $ftc{"$f $e"}++;
    if (exists $dict{$f}->{$e}) {
      print 'T:';
      $txlit++;
    } else { $txlat++; }
    $taps++;
    print $ap;
  }
  print "\n";
}
$txlit /= $taps;
$txlat /= $taps;

my $types = scalar keys %ftc;
my $ttypes = scalar keys %ttc;
my $rt = $ttypes / $types;

print STDERR "     alignments: $taps\n";
print STDERR "transliteration (types): $rt\n";
print STDERR " transliteration (toks): $txlit\n";
print STDERR "     translation (toks): $txlat\n";

