"""Higher-level block factories (encoder/decoder) built atop primitives.
"""
from __future__ import annotations
from typing import List, Tuple
from .primitives import layernorm, mha, ffn, residual_add, Node, cross_attn
from .layout import connect, elbow, Edge


def encoder_block_factory(width_override: float | None = None, three_d: bool = False):
    """Return a builder function for repeat().

    The builder signature: (idx, x, y) -> (nodes, edges, span)
    span: horizontal extent (used by repeat to place next block).
    width is computed from last node's east minus first node's west.
    """

    def build(idx: int, x: float, y: float) -> Tuple[List[Node], List[Edge], float]:
        # Place a canonical encoder sub-sequence starting at (x,y)
        ln1 = layernorm(f"ln{idx}_1", x, y, three_d=three_d)
        att = mha(f"mha{idx}", x + 3.2, y, three_d=three_d)
        add1 = residual_add(f"add{idx}_1", x + 5.6, y, three_d=three_d)
        ln2 = layernorm(f"ln{idx}_2", x + 7.3, y, three_d=three_d)
        ff = ffn(f"ffn{idx}", x + 10.0, y + 0.2, three_d=three_d)
        add2 = residual_add(f"add{idx}_2", x + 12.4, y + 0.2, three_d=three_d)

        nodes = [ln1, att, add1, ln2, ff, add2]
        edges = [
            connect(ln1.anchors["R"], att.anchors["L"]),
            connect(att.anchors["R"], add1.anchors["L"]),
            connect(add1.anchors["R"], ln2.anchors["L"]),
            connect(ln2.anchors["R"], ff.anchors["L"]),
            connect(ff.anchors["R"], add2.anchors["L"]),
            elbow(ln1.anchors["L"], add1.anchors["T"], dx=-0.8),
            elbow(ln2.anchors["L"], add2.anchors["T"], dx=-0.8),
        ]
        span = (add2.x + add2.w) - ln1.x
        if width_override is not None:
            span = width_override
        return nodes, edges, span

    return build

def decoder_block_factory(width_override: float | None = None, include_cross: bool = True, three_d: bool = False):
    """Return builder for decoder block.

    Sequence (simplified): LN -> Masked MHA* -> Add -> LN -> (CrossAttn -> Add ->)? LN -> FFN -> Add
    For layout simplicity we keep linear left-to-right; residual elbows similar to encoder.
    """

    def build(idx: int, x: float, y: float):
        ln1 = layernorm(f"dln{idx}_1", x, y, three_d=three_d)
        mmha = mha(f"dmha{idx}", x + 3.2, y, masked=True, three_d=three_d)
        add1 = residual_add(f"dadd{idx}_1", x + 5.6, y, three_d=three_d)
        nodes = [ln1, mmha, add1]
        edges = [
            connect(ln1.anchors["R"], mmha.anchors["L"]),
            connect(mmha.anchors["R"], add1.anchors["L"]),
            elbow(ln1.anchors["L"], add1.anchors["T"], dx=-0.8),
        ]
        cursor_x = add1.x + 1.7
        if include_cross:
            ln2 = layernorm(f"dln{idx}_2", cursor_x, y, three_d=three_d)
            xatt = cross_attn(f"xatt{idx}", cursor_x + 3.2, y, three_d=three_d)
            add2 = residual_add(f"dadd{idx}_2", cursor_x + 5.6, y, three_d=three_d)
            nodes += [ln2, xatt, add2]
            edges += [
                connect(ln2.anchors["R"], xatt.anchors["L"]),
                connect(xatt.anchors["R"], add2.anchors["L"]),
                elbow(ln2.anchors["L"], add2.anchors["T"], dx=-0.8),
            ]
            cursor_x = add2.x + 1.7
        ln3 = layernorm(f"dln{idx}_3", cursor_x, y, three_d=three_d)
        ff = ffn(f"dffn{idx}", cursor_x + 3.0, y + 0.2, three_d=three_d)
        add3 = residual_add(f"dadd{idx}_3", cursor_x + 5.4, y + 0.2, three_d=three_d)
        nodes += [ln3, ff, add3]
        edges += [
            connect(ln3.anchors["R"], ff.anchors["L"]),
            connect(ff.anchors["R"], add3.anchors["L"]),
            elbow(ln3.anchors["L"], add3.anchors["T"], dx=-0.8),
        ]
        span = (add3.x + add3.w) - ln1.x
        if width_override is not None:
            span = width_override
        return nodes, edges, span

    return build

__all__ = ["encoder_block_factory", "decoder_block_factory"]
