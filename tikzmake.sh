#!/bin/bash

PYTHON=${PYTHON:-python3}

set -euo pipefail

$PYTHON $1.py 
pdflatex $1.tex

rm -f *.aux *.log *.vscodeLog || true
rm -f *.tex || true

if [[ "$OSTYPE" == "darwin"* ]]; then
    open $1.pdf
else
    xdg-open $1.pdf
fi
