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
def bus(src_anchor: str, dst_anchors: Sequence[str], style: str = "conn", stub: float = 0.6, junction: bool = True) -> List[Edge]:
    """Improved fan-out (bus) helper.

    Emits a horizontal stub from src, optional junction marker, then orthogonal elbows
    to each destination anchor. Current export template renders only simple (dx,dy)
    elbows, so we approximate vertical tees by identical via points.

    Args:
      src_anchor: starting anchor string.
      dst_anchors: list of destination anchors.
      style: TikZ edge style key.
      stub: horizontal offset (cm) before branching.
      junction: if True, downstream code may overlay a tiny add node / circle at branch.
    Returns list of Edge objects.
    """
    edges: List[Edge] = []
    for d in dst_anchors:
        edges.append(Edge(src_anchor, d, style, (stub, 0.0)))
    # NOTE: junction rendering (small node) is not represented as an Edge; could be added
    # later by returning a synthetic Node-like object. Kept minimal for now.
    return edges


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


# Stack tag helper ---------------------------------------------------------
def stack_tag(name: str, nodes: Sequence, text: str) -> Any:
    """Return a tiny label node (using same shape semantics as primitives) positioned
    just above the horizontal extent of provided nodes.

    We avoid importing Node directly to keep layout decoupled; assume first node has w/h.
    The caller can inspect returned object's fields similar to primitives.
    """
    if not nodes:
        raise ValueError("nodes sequence empty for stack_tag")
    first = nodes[0]
    last = nodes[-1]
    # Determine bounding box extremes (naive: assume ordered left->right)
    x_center = (first.x + last.x + getattr(last, "w", 0)) / 2.0
    y_top = max(getattr(n, "y", 0) + getattr(n, "h", 0) / 2.0 for n in nodes)
    # Place tag slightly above
    class _Tag:
        def __init__(self, name: str, x: float, y: float, text: str):
            self.name = name
            self.x = x
            self.y = y
            self.w = 1.0
            self.h = 0.5
            self.kind = "tagnode"  # use dedicated style
            self.label = text
            @property
            def anchors(self):  # pragma: no cover - not used currently
                return {"C": f"({self.name}.center)"}
    return _Tag(name, x_center - 0.5, y_top + 0.9, text)
