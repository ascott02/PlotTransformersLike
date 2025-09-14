"""High-level GPT figure helpers.

Provides convenience builders to reduce duplication across example scripts.
"""
from __future__ import annotations
from typing import Tuple, List
from .blocks import decoder_block_factory
from .layout import repeat, group, stack_tag
from .export import export_tex


def build_gpt_stack(n: int = 3, three_d: bool = False, include_cross: bool = False, out_tex: str | None = None):
    """Build a GPT-style decoder stack figure.

    Args:
        n: number of decoder blocks.
        three_d: use 3D primitives.
        include_cross: include cross-attention modules (becomes encoder-decoder style).
        out_tex: optional override output path.
    Returns:
        (out_path, nodes, edges)
    """
    builder = decoder_block_factory(include_cross=include_cross, three_d=three_d)
    nodes, edges = repeat(n, builder, start=(0.0, 0.0), gap=2.0, dir="x")
    g = group("gpt_stack_box", nodes, title=f"Decoder ×{n}{' + Cross' if include_cross else ''}{' (3D)' if three_d else ''}")
    tag = stack_tag("gpt_tag_box", nodes, text=f"$\\times {n}$")
    all_nodes = nodes + [tag]
    if out_tex:
        export_tex(all_nodes, edges, out_tex, boxes=[g])
    return out_tex, all_nodes, edges

__all__ = ["build_gpt_stack"]
