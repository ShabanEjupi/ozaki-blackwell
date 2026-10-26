#!/bin/bash
# build.sh -- main.pdf and the arXiv upload (arxiv.tar.gz: main.tex alone, the bibliography is inline) on the host,
# where docker is; this folder is the same files there (/mnt/data/coder-sister/workspace/ozaki-blackwell/paper).
set -euo pipefail
ssh ampere 'cd /mnt/data/coder-sister/workspace/ozaki-blackwell/paper &&
  docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/w -w /w texlive/texlive:latest-small \
    sh -c "latexmk -pdf -interaction=nonstopmode main.tex >/dev/null; grep -E \"^(Overfull|Underfull|LaTeX Warning|!)\" main.log || true" &&
  latexmk_clean() { rm -f main.aux main.fls main.fdb_latexmk main.log main.out; }; latexmk_clean &&
  tar czf arxiv.tar.gz main.tex && pdfinfo main.pdf 2>/dev/null | grep Pages || ls -la main.pdf'
