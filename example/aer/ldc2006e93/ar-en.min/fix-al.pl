#!/usr/bin/perl -w
use strict;

#<pchunk name="ALJZ_NEWS12_ARB_20050111_133000_67">
#  <s lang="ar">ﻕﺩ ﺖﺟﺮﻳ ﺍﻺﻨﻨﺘﺧﺎﺑﺎﺗ ﻒﻳ ﺐﻴﺗ ﻊﺑﺩ ﺎﻠﻋﺰﻳﺯ ﺎﻠﺤﻜﻴﻣ ﺃﻭ ﻢﺨﺑﺃ ﺎﻠﻨﺟﺭﻮﻤﻨﺘﻳ ﺃﻭ ﺎﻠﻤﻨﻄﻗﺓ ﺎﻠﺨﺿﺭﺍﺀ ، ﻩﺬﻫ ﻕﺩ ﺖﺟﺮﻳ ، ﻮﻗﺩ ﺖﺴﺘﻗﺪﻣ ﺄﻣﺮﻴﻛﺍ ﺈﻠﻫﺍ ، ﺦﻠّﻳ ﻦﻗﻮﻟ ﻢﻠﻳﻮﻧ ﺞﻧﺪﻳ ،</s>
#  <s lang="en">The elections may be held in the house of Abdul Aziz Al - Hakeem or the hideout of Negroponte or the Green Zone , this may be possible , and America might bring in let ' s say a million soldiers .</s>
#  <wordalignment langpair="en-ar">0-2 1-2 2-0 3-1 4-1 5-3 6-4 7-4 8-5 8-6 9-5 9-6 10-5 10-6 11-7 12-7 13-7 14-8 15-9 16-9 17-10 18-10 19-11 20-12 21-13 22-12 23-14 24-15 25-16 26-17 27-17 28-18 29-19 30-21 31-19 32-20 32-22 33-20 33-22 34-24 35-24 36-24 37-25 38-27 39-26 40-27 41-28</wordalignment>

open AR, ">out.ar" or die;
open EN, ">out.en" or die;
open AL, ">out.al" or die;

my ($ar, $en, $al);
while(<>) {
 if (/<pchunk/) {
   $ar = ''; $en = ''; $al ='';
 } elsif (/<\/pchu/) {
   next unless ($ar && $en && $al);
   if ($en =~ /al - | ' (s|ll|t|d|re|ve|m) | (mr|dr|mrs) \. |^(mr|dr|mrs) \. | \% pw /) {
     my @als = split /\s+/, $al;
     my @ens = split /\s+/, $en;
     my @ars = split /\s+/, $ar;
     my %ag;
     for my $ap (@als) {
       my ($e, $a) = split /-/, $ap;
       if (!exists $ag{$e}) { $ag{$e} = []; }
       push @{$ag{$e}}, $a;
     }
     my @re = ();
     my %ra;
     my $elen = scalar @ens;
     my $read_i = 0;
     for (my $i=0; $i<$elen; $i++) {
       if ($i < $elen - 2 && ($ens[$i] eq 'al' && $ens[$i+1] eq '-')) {
         my $new = $ens[$i] . $ens[$i+1] . $ens[$i+2];
         push @re, $new;
         if ($ag{$i}) { for my $fp (@{$ag{$i}}) { $ra{"$read_i-$fp"}=1; } }
         if ($ag{$i+1}) { for my $fp (@{$ag{$i+1}}) { $ra{"$read_i-$fp"}=1; } }
         if ($ag{$i+2}) { for my $fp (@{$ag{$i+2}}) { $ra{"$read_i-$fp"}=1; } }
         $i+=2;
       } elsif ($i < $elen - 1 && (($ens[$i] eq "'" && $ens[$i+1] =~ /^(s|ll|t|d|re|ve|m)$/)
                                  || ($ens[$i] =~ /^(mr|dr|mrs)$/ && $ens[$i+1] eq '.')
				  || ($ens[$i] eq '%' && $ens[$i+1] eq 'pw')     )) {
         my $new = $ens[$i] . $ens[$i+1];
         push @re, $new;
         if ($ag{$i}) { for my $fp (@{$ag{$i}}) { $ra{"$read_i-$fp"}=1; } }
         if ($ag{$i+1}) { for my $fp (@{$ag{$i+1}}) { $ra{"$read_i-$fp"}=1; } }
         $i++;
       } else {
         push @re, $ens[$i];
         if ($ag{$i}) { for my $fp (@{$ag{$i}}) { $ra{"$read_i-$fp"}=1; } }
       }
       $read_i++;
     }
     $al = join ' ', sort keys %ra;
     $en = "@re";
   }
   print AR "$ar\n";
   print EN "$en\n";
   print AL "$al\n";
 } elsif (/<s lang="ar">(.*)<\/s>/) {
   $ar = lc $1;
 } elsif (/<s lang="en">(.*)<\/s>/) {
   $en = lc $1;
 } elsif (/<wordalignment langpair="en-ar">(.*)<\/w/) {
   $al = $1;
 }
}

close AR;
close EN;
close AL;

