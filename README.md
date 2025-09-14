# PlotNeuralNet
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2526396.svg)](https://doi.org/10.5281/zenodo.2526396)

Latex code for drawing neural networks for reports and presentation. Have a look into examples to see how they are made. Additionally, lets consolidate any improvements that you make and fix any bugs to help more people with this code.

## Examples

Following are some network representations:

<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308846-c2231880-049c-11e9-8763-3daa1024de78.png" width="85%" height="85%"></p>
<h6 align="center">FCN-8 (<a href="https://www.overleaf.com/read/kkqntfxnvbsk">view on Overleaf</a>)</h6>


<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308873-e2eb6e00-049c-11e9-9587-9da6bdec011b.png" width="85%" height="85%"></p>
<h6 align="center">FCN-32 (<a href="https://www.overleaf.com/read/wsxpmkqvjnbs">view on Overleaf</a>)</h6>


<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308911-03b3c380-049d-11e9-92d9-ce15669017ad.png" width="85%" height="85%"></p>
<h6 align="center">Holistically-Nested Edge Detection (<a href="https://www.overleaf.com/read/jxhnkcnwhfxp">view on Overleaf</a>)</h6>

## Getting Started
1. Install the following packages on Ubuntu.
    * Ubuntu 16.04
        ```
        sudo apt-get install texlive-latex-extra
        ```

    * Ubuntu 18.04.2
Base on this [website](https://gist.github.com/rain1024/98dd5e2c6c8c28f9ea9d), please install the following packages.
        ```
        sudo apt-get install texlive-latex-base
        sudo apt-get install texlive-fonts-recommended
        sudo apt-get install texlive-fonts-extra
        sudo apt-get install texlive-latex-extra
        ```

    * Windows
    1. Download and install [MikTeX](https://miktex.org/download).
    2. Download and install bash runner on Windows, recommends [Git bash](https://git-scm.com/download/win) or Cygwin(https://www.cygwin.com/)

2. Execute the example as followed.
    ```
    cd pyexamples/
    bash ../tikzmake.sh test_simple
    ```

## TODO

- [X] Python interface
- [ ] Add easy legend functionality
- [ ] Add more layer shapes like TruncatedPyramid, 2DSheet etc
- [ ] Add examples for RNN and likes.

## Latex usage

See [`examples`](examples) directory for usage.

## Python usage

First, create a new directory and a new Python file:

    $ mkdir my_project
    $ cd my_project
    vim my_arch.py

Add the following code to your new file:

```python
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    to_connection( "pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    to_connection("pool2", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
```

Now, run the program as follows:

    bash ../tikzmake.sh my_arch

## Transformer Extensions (Experimental)

This fork adds a lightweight Python → TikZ path for Transformer-era diagrams (ViT / GPT / Encoder–Decoder) located in `plotnn_xt/` with examples in `examples/`:

Current example figures:

* `fig_encoder_block.tex` – single encoder block (LN → MHA → Add → LN → FFN → Add)
* `fig_decoder_block.tex` – decoder block with masked self-attn + cross-attn
* `fig_encoder_stack.tex` – vertical stack of encoder blocks (repeat helper)
* `fig_gpt_stack.tex` – horizontal causal decoder stack (no cross-attn)
* `fig_vit_patchflow.tex` – ViT patch embed → [CLS]+PosEnc → N× encoders → CLS head
* `fig_encdec_overview.tex` – encoder stack feeding decoder stack with cross-attn edges

Build all example PDFs:

```
make examples
```

Each `.tex` lives beside its generated `.pdf` under `examples/`.

### SVG Export

If `pdf2svg` is installed the Makefile offers bulk conversion:

```
make svg_all
```

Exports land in `examples/assets/*.svg`. Single file:

```
make examples/assets/fig_vit_patchflow.svg
```

If `pdf2svg` is missing, the target prints a helpful message. Alternative: use `dvisvgm` by modifying the Makefile (not yet included).

### Adding New Primitives

See `plotnn_xt/primitives.py` for helper constructors returning `Node` objects with named anchors (`L,R,T,B,C`). Add a new function returning a `Node` and then reference it in an example, finally rebuild with `make`.

### Testing

Minimal emission tests live in `tests/test_emit_tex.py` (pytest). Run:

```
pytest -q
```

These assert that key labels (e.g., `MHA*`, `CrossAttn`) render into emitted `.tex`.

---
Status: experimental; styles in `transformer_tex/transformer_styles.tex` will evolve (color themes, tag node styling, etc.).


### Legacy Compatibility (Original PlotNeuralNet Style)

If you prefer (or need to keep) the original list‑of‑strings workflow using `pycore/tikzeng.py`, a thin compatibility layer now lets you author simple Transformer blocks without switching to the new object model immediately.

Key additions in `pycore/tikzeng.py`:

* `to_LayerNorm(name, x, y, ...)`
* `to_MHA(name, x, y, heads=8, d_model=768, masked=False)` (adds `*` in label if masked)
* `to_FFN(name, x, y, dff=3072)`
* `to_Add(name, x, y)` residual add node (`+$+$` circle)
* `to_CLSHead(name, x, y, classes=1000)` classification head
* `to_transformer_block(prefix, x, y)` convenience macro emitting a minimal encoder‑style micro block: LN → MHA → LN → FFN → Add

Minimal legacy example (`examples/fig_gpt_stack_legacy.py`):

```python
from pycore import tikzeng as T

arch = [
    T.to_head('..'),
    T.to_begin(),
]
x0 = 0.0
y0 = 0.0
for i in range(3):
        arch.extend(T.to_transformer_block(f"b{i}_", x0, y0))
        x0 += 18.0  # horizontal spacing per block
arch.append(T.to_end())
T.to_generate(arch, 'examples/fig_gpt_stack_legacy.tex')
```

Build (identical to original):

```
latexmk -pdf examples/fig_gpt_stack_legacy.tex
```

When to use legacy mode:

* You already have older CNN scripts and want to sprinkle in Transformer elements fast.
* Teaching / quick sketches where full node/edge composition is overkill.

When to migrate to the new object API (`plotnn_xt/`):

* You need grouping boxes, stack tags, or cross‑attn elbows.
* You want tests around emitted diagrams.
* You plan iterative styling (themes, dark mode) without rewriting every script.

Migration path:
1. Replace legacy conv snippets with `to_transformer_block` to confirm layout.
2. Transition to `plotnn_xt.primitives` + `export_tex` for richer figures.
3. Drop string concatenation and use structured nodes (`Node`, `Edge`, `Box`).

Limitations (legacy layer):
* No automatic group boxes or lanes.
* No edge helper macros yet (connections still manual if desired).
* Styling updates (themes) applied globally via `transformer_styles.tex`—object model offers finer future control.

If you need legacy edge helpers or a migration cheat‑sheet, open an issue or extend the TODOs (see below).



