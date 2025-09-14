"""Layout helpers for transformer diagrams."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, List, Sequence, Callable, Any


@dataclass
class Edge:
    src: str  # anchor expression or (node.anchor)
    dst: str
    style: str = "conn"
    via: Optional[Tuple[float, float]] = None  # (dx, dy) waypoint relative from src


@dataclass
class Box:
    name: str
    nodes: Sequence
    kind: str = "gbox"  # style for group box
    title: str | None = None

    def tikz_fit_expr(self) -> str:
        names = " ".join([f"({n.name})" for n in self.nodes])
        return names


def connect(a: str, b: str, style: str = "conn") -> Edge:
    return Edge(a, b, style)


def elbow(a: str, b: str, dx: float = 0.8, dy: Optional[float] = None, style: str = "conn") -> Edge:
    return Edge(a, b, style, (dx, dy if dy is not None else 0.0))


# Simple horizontal stacking ------------------------------------------------

def stack_x(nodes: Iterable, start=(0.0, 0.0), gap: float = 0.8) -> List:
    x, y = start
    placed = []
    for n in nodes:
        # copy with new x/y
        new = n.__class__(**{**n.__dict__, "x": x, "y": y})  # type: ignore[arg-type]
        placed.append(new)
        x += getattr(n, "w", 1.0) + gap
    return placed


# Group / lane convenience -------------------------------------------------
def group(name: str, nodes: Sequence, title: str | None = None) -> Box:
    return Box(name=name, nodes=nodes, title=title, kind="gbox")


def lane(name: str, nodes: Sequence, title: str | None = None) -> Box:
    return Box(name=name, nodes=nodes, title=title, kind="lane")


# Bus / fan-out -------------------------------------------------------------
def bus(src_anchor: str, dst_anchors: Sequence[str], style: str = "conn", stub: float = 0.6) -> List[Edge]:
    """Create a simple tee fan-out: horizontal stub then vertical to each dst.
    For now we emit separate elbow edges with same dx stub.
    """
    return [Edge(src_anchor, d, style, (stub, 0.0)) for d in dst_anchors]


# Repeat / stacks ----------------------------------------------------------
def repeat(n: int, block_fn: Callable[[int, float, float], Tuple[List[Any], List[Edge], float]], start=(0.0, 0.0), gap: float = 1.0, dir: str = "x") -> Tuple[List[Any], List[Edge]]:
    """Repeat a block builder n times along an axis.

    block_fn: (idx, x, y) -> (nodes, edges, span)
        span: advance amount along the chosen axis (width consumed by block)
    dir: 'x' or 'y'
    Returns all nodes and edges concatenated.
    """
    assert dir in {"x", "y"}
    ox, oy = start
    all_nodes: List[Any] = []
    all_edges: List[Edge] = []
    for i in range(n):
        nodes, edges, span = block_fn(i, ox, oy)
        all_nodes.extend(nodes)
        all_edges.extend(edges)
        if dir == "x":
            ox += span + gap
        else:
            oy -= span + gap  # vertical stacking downward by default
    return all_nodes, all_edges
