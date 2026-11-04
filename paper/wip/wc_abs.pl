#!/usr/bin/perl
# wc_abs.pl main.tex -- words of the abstract as a reader counts them: TeX commands dropped, math kept as one token per run
local $/; my $t = <>; $t =~ /\\begin\{abstract\}(.*?)\\end\{abstract\}/s or die "no abstract\n"; my $a = $1;
$a =~ s/\\texttt\{([^}]*)\}/$1/g; $a =~ s/\\(emph|textbf|mathbf|mathrm)\{([^}]*)\}/$2/g; $a =~ s/\\[a-zA-Z]+//g;
$a =~ s/[{}\\]//g; $a =~ s/~/ /g; my @w = grep { /[A-Za-z0-9]/ } split /\s+/, $a; print scalar(@w), " words\n";
