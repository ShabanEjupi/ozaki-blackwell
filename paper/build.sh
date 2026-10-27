#!/bin/bash
# build.sh -- main.pdf and the arXiv upload (arxiv.tar.gz: main.tex alone, the bibliography is inline) on the host,
# where docker is; this folder is the same files there (/mnt/data/coder-sister/workspace/ozaki-blackwell/paper).
set -euo pipefail
# 27-10-2026: three LaTeX phrases were broken in one day by perl replacements ($4 and $K read as perl variables): a
# control byte or a "sim~" without its backslash stops the build here.
cd "$(dirname "$0")"
if grep -nP "[\x00-\x08\x0b-\x1f]" main.tex || grep -nP "(?<!\\\\)sim~" main.tex; then echo "main.tex damaged (above)"; exit 1; fi
ssh ampere 'cd /mnt/data/coder-sister/workspace/ozaki-blackwell/paper &&
  docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/w -w /w texlive/texlive:latest-small \
    sh -c "latexmk -pdf -interaction=nonstopmode main.tex >/dev/null; grep -E \"^(Overfull|Underfull|LaTeX Warning|!)\" main.log || true" &&
  latexmk_clean() { rm -f main.aux main.fls main.fdb_latexmk main.log main.out; }; latexmk_clean &&
  tar czf arxiv.tar.gz main.tex && pdfinfo main.pdf 2>/dev/null | grep Pages || ls -la main.pdf'
