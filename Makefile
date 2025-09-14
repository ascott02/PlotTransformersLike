LATEXMK=latexmk -halt-on-error -quiet
PYTHON?=python3

# Python-driven example scripts that emit .tex when executed
PY_EX_SCRIPTS=\
  examples/fig_encoder_block.py \
  examples/fig_decoder_block.py \
  examples/fig_encoder_stack.py \
  examples/fig_vit_patchflow.py \
  examples/fig_encdec_overview.py
	examples/fig_vit_patchflow.py \
	examples/fig_encdec_overview.py \
	$(wildcard examples/gpt/*.py)

TEX_FROM_PY=$(PY_EX_SCRIPTS:.py=.tex)
PDFS=$(TEX_FROM_PY:.tex=.pdf)
SVGS=$(PDFS:examples/%.pdf=examples/assets/%.svg)

all: examples

examples: $(PDFS)

# Run python script to (re)generate its .tex output
examples/%.tex: examples/%.py transformer_tex/transformer_styles.tex
	$(PYTHON) $< >/dev/null 2>&1 || true

examples/%.pdf: examples/%.tex transformer_tex/transformer_styles.tex
	$(LATEXMK) -pdf -outdir=examples $<

svg: examples/assets/fig_encoder_block.svg

svg_all: $(SVGS)

examples/assets/%.svg: examples/%.pdf | examples/assets
	@if command -v pdf2svg >/dev/null 2>&1; then \
	  pdf2svg $< $@; \
	else \
	  echo "pdf2svg not installed; skipping $@"; \
	fi

examples/assets:
	mkdir -p examples/assets

clean:
	latexmk -C
	rm -f examples/assets/*.svg || true

.PHONY: all examples svg svg_all clean
