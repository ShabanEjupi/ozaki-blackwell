#!/usr/bin/perl
# rep.pl FILE OLD NEW -- replaces the one literal occurrence of OLD's text by NEW's text in FILE; refuses 0 or 2+ occurrences
use strict; my ($f, $o, $n) = @ARGV;
local $/; open my $F, '<:raw', $f or die "$f: $!"; my $t = <$F>; close $F;
open my $O, '<:raw', $o or die; my $os = <$O>; close $O; open my $N, '<:raw', $n or die; my $ns = <$N>; close $N;
chomp $os; chomp $ns;
my $c = () = $t =~ /\Q$os\E/g; die "rep.pl: $c occurrences in $f\n" unless $c == 1;
$t =~ s/\Q$os\E/$ns/; open $F, '>:raw', $f or die; print $F $t; close $F; print "replaced in $f\n";
