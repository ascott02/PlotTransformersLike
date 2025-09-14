"""Transformer-era primitives for PlotNeuralNet extension.
Initial bootstrap version.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class Node:
    name: str
    x: float
    y: float
    w: float
    h: float
    kind: str  # tikz style key: blk / sblk / addnode etc.
    label: str = ""

    @property
    def anchors(self) -> Dict[str, str]:
        return {
            "L": f"({self.name}.west)",
            "R": f"({self.name}.east)",
            "T": f"({self.name}.north)",
            "B": f"({self.name}.south)",
            "C": f"({self.name}.center)",
        }


# Core transformer pieces --------------------------------------------------

def layernorm(name: str, x: float, y: float, w: float = 2.6, h: float = 0.6, label: str = "LayerNorm", three_d: bool = False) -> Node:
    kind = "sblk3d" if three_d else "sblk"
    return Node(name, x, y, w, h, kind, label)


def ffn(name: str, x: float, y: float, w: float = 3.2, h: float = 1.2, dff: int = 3072, three_d: bool = False) -> Node:
    kind = "blk3d" if three_d else "blk"
    return Node(name, x, y, w, h, kind, f"FFN\\\\\\scriptsize($d_{{ff}}={dff}$)")


def mha(name: str, x: float, y: float, w: float = 3.8, h: float = 1.2, heads: int = 8, d_model: int = 768, masked: bool = False, three_d: bool = False) -> Node:
    """Multi-head attention primitive.

    If masked=True a trailing "*" is appended to label to visually denote causal/masked attention.
    Future: could swap style/color or add a corner tag.
    """
    mask = "*" if masked else ""
    kind = "blk3d" if three_d else "blk"
    return Node(name, x, y, w, h, kind, f"MHA{mask}\\\\\\scriptsize(h={heads}, $d_{{model}}={d_model}$)")


def cross_attn(name: str, x: float, y: float, w: float = 3.8, h: float = 1.2, heads: int = 8, d_model: int = 768, three_d: bool = False) -> Node:
    """Cross-attention primitive (encoder-decoder)."""
    kind = "blk3d" if three_d else "blk"
    return Node(name, x, y, w, h, kind, f"CrossAttn\\\\\\scriptsize(h={heads}, $d_{{model}}={d_model}$)")


def residual_add(name: str, x: float, y: float, r: float = 0.22, three_d: bool = False) -> Node:
    kind = "addnode3d" if three_d else "addnode"
    return Node(name, x, y, r, r, kind, "+")


# ViT specific --------------------------------------------------------------

def patch_embed(name: str, x: float, y: float, w: float = 3.2, h: float = 1.0, patch: str = "16×16", d: int = 768, three_d: bool = False) -> Node:
    kind = "blk3d" if three_d else "blk"
    return Node(name, x, y, w, h, kind, f"PatchEmbed\\\\\\scriptsize({patch}→$d={d}$)")


def class_token(name: str, x: float, y: float, w: float = 1.4, h: float = 0.8, three_d: bool = False) -> Node:
    kind = "sblk3d" if three_d else "sblk"
    return Node(name, x, y, w, h, kind, "[CLS]")


def pos_enc(name: str, x: float, y: float, w: float = 1.6, h: float = 0.8, three_d: bool = False) -> Node:
    kind = "sblk3d" if three_d else "sblk"
    return Node(name, x, y, w, h, kind, "PosEnc")


# Small badges / auxiliaries ------------------------------------------------
def dropout(name: str, x: float, y: float, w: float = 1.0, h: float = 0.5, p: float = 0.1, three_d: bool = False) -> Node:
    kind = "sblk3d" if three_d else "sblk"
    return Node(name, x, y, w, h, kind, f"Dropout\\\\\\scriptsize(p={p})")


# Projection heads ----------------------------------------------------------
def cls_head(name: str, x: float, y: float, w: float = 2.4, h: float = 1.0, n_classes: int = 1000, three_d: bool = False) -> Node:
    """Classification head (dense + softmax) placeholder.

    Rendered as a standard block with class count hint. Downstream could add
    LM heads similarly (tie weights etc.) but for figure purposes we only show dimensions.
    """
    kind = "blk3d" if three_d else "blk"
    return Node(name, x, y, w, h, kind, f"CLS Head\\\\\\scriptsize($C={n_classes}$)")
