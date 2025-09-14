GPT Examples
============

Files:
  fig_gpt_stack.py       - Standard (2D) decoder-only stack (no cross-attn)
  fig_gpt_stack_3d.py    - 3D styled decoder-only stack
  fig_gpt_stack_legacy.py- Legacy macro-based variant

Helper API:
  plotnn_xt.gpt.build_gpt_stack(n, three_d=False, include_cross=False, out_tex=...)

Usage:
  python examples/gpt/fig_gpt_stack.py
  latexmk -pdf examples/gpt/fig_gpt_stack.tex

Switch to 3D globally for GPT examples by calling build_gpt_stack(..., three_d=True).

To add cross-attention (turning each decoder block into part of an encoder-decoder):
  build_gpt_stack(n=4, include_cross=True, out_tex='examples/gpt/fig_gpt_stack_cross.tex')

The 3D effect comes from TikZ styles blk3d/sblk3d/addnode3d in transformer_styles.tex.
