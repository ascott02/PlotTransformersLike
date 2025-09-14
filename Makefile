LATEXMK=latexmk -halt-on-error -quiet

EXAMPLES=examples/fig_encoder_block.pdf

all: examples

examples: $(EXAMPLES)

examples/%.pdf: examples/%.tex transformer_tex/transformer_styles.tex
	$(LATEXMK) -pdf $<

svg: examples/assets/fig_encoder_block.svg

examples/assets/%.svg: examples/%.pdf
	pdf2svg $< $@

clean:
	latexmk -C
	rm -f examples/assets/*.svg || true

.PHONY: all examples svg clean
