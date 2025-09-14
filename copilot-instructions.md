# PlotNeuralNet → Transformers (ViTs/GPTs) — Terraforming README

> This README is a **step‑by‑step prompt** to scaffold a fork of \[HarisIqbal88/PlotNeuralNet] and extend it with **Transformer‑era primitives** (ViTs/GPTs/TimeSformer), keeping the original “**Python → TikZ**” workflow. Follow the checklists top‑to‑bottom to get to usable figures quickly. This directory is a fork of the official repository. 

---

## Mission & scope

**Goal:** Produce **paper‑grade** Transformer diagrams from **Python code** using **TikZ**, with reusable building blocks and minimal boilerplate.

**We will add:**

* Primitives: **MHA**, **Masked MHA**, **FFN**, **LayerNorm**, **Residual (+)**, **Dropout** badges, **Embeddings** (token/positional), **PatchEmbed** (ViT), **ClassToken**, **Projection heads** (LM head, CLS head), **Concat/Split**, **Cross‑Attention** (for encoder‑decoder & VLMs), **Modality adapters**.
* Layout helpers: **ports** (`.L/.R/.T/.B/.C`), **elbow/waypoint routing**, **fan‑out buses**, **stacks/repeaters** (`×N`), **groups/fit boxes** with titles, **lanes** (e.g., text vs image).
* Export: clean **standalone TikZ** (`.tex`) + optional **PDF/SVG** for slide decks (Marp).

**Non‑goals:** Full auto‑layout; arbitrary spline routing. We prioritize explicit, deterministic layouts with ergonomic helpers.

---

## Repository layout (target)

```
plotnn-transformers/
├─ upstream/                         # your fork/checkout of PlotNeuralNet (or submodule)
├─ plotnn_xt/                        # new extension code (Python)
│  ├─ __init__.py
│  ├─ primitives.py                  # MHA, FFN, LN, PatchEmbed, Residual, Proj, etc.
│  ├─ layout.py                      # ports, elbows, bus/fanout, stacks, repeaters
│  ├─ export.py                      # Jinja/TikZ emitter + wrappers
│  └─ cli.py                         # optional: `plotnn-tx build examples/fig_*`
├─ transformer_tex/
│  ├─ transformer_styles.tex         # TikZ styles, colors, arrowheads
│  └─ transformer_macros.tex         # (optional) pure TikZ macros for LaTeX-only users
├─ examples/
│  ├─ fig_encoder_block.py           # LN→MHA(+res)→LN→FFN(+res)
│  ├─ fig_decoder_block.py           # Masked MHA + (Cross‑Attn) + FFN
│  ├─ fig_gpt_stack.py               # N× decoder layers (GPT‑style)
│  ├─ fig_vit_patchflow.py           # Patchify→[CLS]→PosEnc→N× encoders
│  ├─ fig_timesformer_axes.py        # factorized space/time attention lanes
│  └─ assets/                        # built PDFs/SVGs for quick browsing
├─ tests/
│  ├─ test_emit_tex.py               # unit tests on the emitted TikZ
│  └─ test_render_regression.py      # optional: image diff regression
├─ Makefile
├─ requirements.txt
└─ README.md
```

---

## Bootstrap — 15‑minute path

* [ ] **Fork** PlotNeuralNet and clone into `upstream/` (or add as submodule).
* [ ] Create the directories above.
* [ ] `python -m venv .venv && source .venv/bin/activate`
* [ ] `pip install -r requirements.txt` (see minimal list below).
* [ ] Ensure TeX Live (2021+) with `tikz`, `standalone`, `pgf`, `xcolor`, `calc`, `arrows.meta`.
* [ ] `make examples` → build the starter figures.

**requirements.txt (minimal):**

```
Jinja2>=3.1
click>=8.1     # if using the CLI
rich>=13       # nice terminal output (optional)
Pillow>=10     # only if doing render regression tests
```

---

## Design principles

1. **Deterministic layout**: authors place blocks explicitly; helpers reduce tedium.
2. **Named ports/anchors**: consistent connections (`.east→.west`, residuals via top lanes).
3. **Separation of concerns**: Python composes; TikZ styles centralize appearance.
4. **Tiny, composable primitives** over giant monolithic macros.
5. **One‑command build** for examples, plus CI check.
## Bootstrap — 15‑minute path (Progress)

Status: initial extension scaffolding complete; encoder block example and unit test pass.

* [ ] **Fork** PlotNeuralNet and clone into `upstream/` (or add as submodule). *(pending explicit upstream checkout)*
* [x] Create the directories above (`plotnn_xt/`, `transformer_tex/`, `examples/assets/`, `tests/`).
* [x] `python -m venv .venv && source .venv/bin/activate`
* [x] `pip install -r requirements.txt`
* [ ] Ensure TeX Live (2021+) with required packages. *(local environment assumed; not yet validated in CI)*
* [x] `make examples` → builds starter encoder block figure.

## Python API (current state)

Implemented primitives expose named anchors (`L,R,T,B,C`). Masked MHA indicated with a trailing `*` in label. Added `dropout` badge.

## TikZ styles (transformer\_styles.tex)
```python
from dataclasses import dataclass

@dataclass
class Node:
    name: str; x: float; y: float; w: float; h: float; kind: str; label: str = ""
    @property
    def anchors(self):
        return {k: f"({self.name}.{v})" for k, v in {"L":"west","R":"east","T":"north","B":"south","C":"center"}.items()}

def layernorm(...); def ffn(...); def mha(..., masked=False); def residual_add(...)
def patch_embed(...); def class_token(...); def pos_enc(...); def dropout(...)
```
class Node:
## TikZ styles (transformer\_styles.tex)
Added `gbox` style for grouping boxes. (Styles live in `transformer_tex/transformer_styles.tex`).
  @property
  def anchors(self):
    return {
      "L": f"({self.name}.west)", "R": f"({self.name}.east)",
## Groups, lanes, buses (current helpers)
Implemented API (layout.py): `group(name, nodes, title=None)`, `lane(name, nodes, title=None)`, and `bus(src_anchor, [dst_anchors], stub=0.6)` returning edges with a horizontal stub.

Rendering of group/lane boxes into TikZ not yet added to template (next task).
      "C": f"({self.name}.center)",
**Phase 0 — Bootstrap**
* [ ] Repo layout created; upstream fork checked out (local fork structure partially present, upstream submodule not added)
* [x] `transformer_styles.tex` compiled in a smoke test (encoder block PDF builds)

**Phase 1 — Core primitives**
* [x] `LayerNorm`, `FFN`, `MHA` (+ masked flag via label asterisk)
* [x] `ResidualAdd (+)`, `Dropout` badges

**Phase 2 — Helpers**
* [x] Ports API; `connect` and `elbow` with waypoints
* [x] `group(...)` + `lane(...)` data structures (template integration pending)
* [x] `bus(...)` fan‑out helper (edge list) — needs visual refinement
  return Node(name,x,y,w,h,"blk", f"FFN\\\\\\scriptsize($d_{{ff}}={dff}$)")
**Phase 3 — Stacks**
* [ ] `repeat(n, block, dir, gap)` utility with `×n` label
* [ ] Encoder stack example; GPT stack example
def mha(name, x, y, w=3.8, h=1.2, heads=8, d_model=768):
**Phase 4 — ViT**
* [x] `PatchEmbed`, `ClassToken`, `PosEnc` primitives
* [ ] End‑to‑end ViT figure

**Phase 5 — Decoder/Cross‑Attn**
* [ ] Decoder block with masked MHA + cross‑attn
* [ ] Encoder–decoder figure
def patch_embed(name,x,y,w=3.2,h=1.0,patch="16×16",d=768):
**Phase 6 — Exports**
* [ ] SVG export path documented & tested (Makefile target exists; need doc/test)
* [ ] Marp slide embed demo

**Phase 7 — Polish**
* [ ] Theme presets (paper vs slides/dark)
* [ ] Example gallery README
* [ ] Render regression test (placeholder only)

## Next Immediate Tasks

1. Integrate group/lane/box rendering in export template (draw fit rectangles + optional title tag).
2. Implement `repeat(n, block_fn, dir='x')` and example for encoder stack.
3. Create decoder block primitive composition (masked MHA + cross-attn placeholder) and example script.
4. ViT end-to-end figure (patch→cls concat→pos enc→stack×N→cls head placeholder).

def pos_enc(name,x,y,w=1.6,h=0.8):   return Node(name,x,y,w,h,"sblk","PosEnc")

def residual_add(name,x,y,r=0.22):   return Node(name,x,y,r,r,"add","+")
```

Routing & helpers:

```python
# plotnn_xt/layout.py
from dataclasses import dataclass
@dataclass
class Edge: src: str; dst: str; style: str = "conn"; via: tuple|None = None

def connect(a,b,style="conn"): return Edge(a,b,style)

def elbow(a,b,dx=0.8,dy=None,style="conn"): return Edge(a,b,style,(dx,dy))

def stack_x(nodes, start=(0.0,0.0), gap=0.8):
  x,y = start; placed=[]
  for n in nodes: placed.append(n.__class__(**{**n.__dict__,"x":x,"y":y})); x += n.w + gap
  return placed
```

Emitter (Jinja → standalone TikZ):

```python
# plotnn_xt/export.py
from pathlib import Path
from jinja2 import Template
TPL = r"""
\documentclass[tikz,border=2pt]{standalone}
\input{transformer_tex/transformer_styles.tex}
\begin{document}
\begin{tikzpicture}
{% for n in nodes %}
  \node[{{n.kind}}={{n.w}}cm/{{n.h}}cm] ({{n.name}}) at ({{n.x}}cm,{{n.y}}cm) { {{n.label}} };
{% endfor %}
{% for e in edges %}
  {% if e.via %}\path[{{e.style}}] {{e.src}} -- ++({{e.via[0]}}cm,0) |- {{e.dst}};{% else %}
  \path[{{e.style}}] {{e.src}} -- {{e.dst}};{% endif %}
{% endfor %}
\end{tikzpicture}
\end{document}
"""

def export_tex(nodes, edges, out_tex): Path(out_tex).write_text(Template(TPL).render(nodes=nodes,edges=edges)); return out_tex
```

---

## Example: Encoder block (first win)

```python
# examples/fig_encoder_block.py
from plotnn_xt.primitives import layernorm, mha, ffn, residual_add
from plotnn_xt.export import export_tex
from plotnn_xt.layout import connect, elbow

ln1 = layernorm("ln1", 0, 0)
att = mha("mha", 3.2, 0)
add1= residual_add("add1", 5.6, 0)
ln2 = layernorm("ln2", 7.3, 0)
ff  = ffn("ffn", 10.0, 0.2)
add2= residual_add("add2", 12.4, 0.2)

nodes=[ln1,att,add1,ln2,ff,add2]
edges=[
  connect(ln1.anchors["R"], att.anchors["L"]),
  connect(att.anchors["R"], add1.anchors["L"]),
  connect(add1.anchors["R"], ln2.anchors["L"]),
  connect(ln2.anchors["R"], ff.anchors["L"]),
  connect(ff.anchors["R"], add2.anchors["L"]),
  elbow(ln1.anchors["L"], add1.anchors["T"], dx=-0.8),
  elbow(ln2.anchors["L"], add2.anchors["T"], dx=-0.8),
]
export_tex(nodes,edges,"examples/fig_encoder_block.tex")
```

Build:

```bash
latexmk -pdf examples/fig_encoder_block.tex
```

---

## ViT & GPT specifics (what to implement next)

* **ViT**: `PatchEmbed` → `[CLS]` concat → `PosEnc` → `repeat N × EncoderBlock` → `CLS head`.
* **GPT**: `repeat N × DecoderBlock(masked MHA → Add → LN → MHA? → FFN)` with **mask badge** on MHA; **LM head** projection at output.
* **Cross‑Attention**: decoder block includes cross‑attn fed by encoder lane (for encoder‑decoder models).
* **TimeSformer**: add **lanes** for **space** vs **time** attention; use dashed lane boxes and labels.

---

## Groups, lanes, buses

* `group(name, nodes..., title="Encoder Block")` → draws a rounded rectangle around members with a header label.
* `lane(title, nodes..., style=lane)` → visual separation (e.g., *Text*, *Vision*).
* `bus(src.R -> {dst1.L, dst2.L, dst3.L})` → fan‑out helper that emits a tee split with short stubs.

(These are convenience APIs on top of `layout.py`; implement progressively.)

---

## Export to SVG (for Marp slides)

Option A (PDF → SVG via **pdf2svg**):

```bash
latexmk -pdf examples/fig_encoder_block.tex
pdf2svg examples/fig_encoder_block.pdf examples/assets/fig_encoder_block.svg
```

Option B (DVI → SVG via **dvisvgm**):

```bash
latexmk -latex=latex -dvi examples/fig_encoder_block.tex
dvisvgm --no-fonts -o examples/assets/fig_encoder_block.svg examples/fig_encoder_block.dvi
```

Embed in Marp:

```markdown
![](examples/assets/fig_encoder_block.svg)
```

---

## Makefile (starter)

```make
LATEXMK=latexmk -halt-on-error -quiet

examples: examples/fig_encoder_block.pdf

examples/%.pdf: examples/%.tex transformer_tex/transformer_styles.tex
	$(LATEXMK) -pdf $<

svg: examples/assets/fig_encoder_block.svg

examples/assets/%.svg: examples/%.pdf
	pdf2svg $< $@

clean:
	latexmk -C
	rm -f examples/assets/*.svg
```

---

## Testing & CI (optional but recommended)

* **Unit (emit)**: assert node names/labels/anchors appear in `.tex`.
* **Render regression**: build PDF → PNG (e.g., `magick -density 300`) → compare with baseline using `Pillow`.
* **GitHub Actions**: set up TeX Live + Python; build `examples/` on PRs.

---

## Roadmap — implementation checklist

**Phase 0 — Bootstrap**

* [ ] Repo layout created; upstream fork checked out
* [ ] `transformer_styles.tex` compiled in a smoke test

**Phase 1 — Core primitives**

* [ ] `LayerNorm`, `FFN`, `MHA` (+ masked flag)
* [ ] `ResidualAdd (+)`, `Dropout` badges

**Phase 2 — Helpers**

* [ ] Ports API; `connect` and `elbow` with waypoints
* [ ] `group(...)` fit boxes; `lane(...)` overlays
* [ ] `bus(...)` fan‑out helper

**Phase 3 — Stacks**

* [ ] `repeat(n, block, dir, gap)` utility with `×n` label
* [ ] Encoder stack example; GPT stack example

**Phase 4 — ViT**

* [ ] `PatchEmbed`, `ClassToken`, `PosEnc` primitives
* [ ] End‑to‑end ViT figure

**Phase 5 — Decoder/Cross‑Attn**

* [ ] Decoder block with masked MHA + cross‑attn
* [ ] Encoder–decoder figure

**Phase 6 — Exports**

* [ ] SVG export path documented & tested
* [ ] Marp slide embed demo

**Phase 7 — Polish**

* [ ] Theme presets (paper vs slides/dark)
* [ ] Example gallery README

---

## Style guide (visual + naming)

* Connect **left→right**; prefer top‑lane residuals.
* Keep parameter hints **inside nodes** with `\scriptsize`.
* Use math for subscripts: `$d_{ff}$`, `$d_{model}$`.
* Node names: short (`ln1`, `mha2`, `ff3`, `patch`, `cls`, `pos`).
* Keep lanes clearly separated when mixing modalities.

---

## Attribution & license

Retain PlotNeuralNet’s original license and attribution. Clearly document new files and authorship. If distributing examples, ensure any external assets are licensed appropriately.

---

## FAQ

**Why not Mermaid/D2?** Great for quick class sketches; this repo targets **publication‑grade** LaTeX figures.

**Auto‑layout?** Out of scope; helpers provide repeaters, lanes, buses, and elbow routing that is predictable.

**Can this read real models?** Yes—add a small JSON schema (modules + edges) exported from PyTorch and map to primitives + layout policy.

---

**You’re ready.** Create the directories, paste the starter files, and build the first encoder block. Iterate by adding primitives and helpers as you need them.
