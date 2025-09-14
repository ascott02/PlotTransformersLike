"""Export utilities (Jinja -> standalone TikZ).

Enhancements:
  * Support rendering of grouping / lane boxes (``layout.Box``) using TikZ ``fit``.
  * Boxes are drawn on the background layer so they appear behind primitives.
"""
from __future__ import annotations
from pathlib import Path
from jinja2 import Template
from typing import Sequence, Optional

try:  # circular-safe import (only needed for type hints / template attrs)
  from .layout import Box  # type: ignore
except Exception:  # pragma: no cover
  Box = None  # type: ignore

TPL = r"""
\documentclass[tikz,border=2pt]{standalone}
\input{transformer_tex/transformer_styles.tex}
\begin{document}
\begin{tikzpicture}
% --- Primitive nodes -------------------------------------------------------
{% for n in nodes %}
  \node[{{n.kind}}={{'%.2f'%n.w}}cm/{{'%.2f'%n.h}}cm] ({{n.name}}) at ({{'%.2f'%n.x}}cm,{{'%.2f'%n.y}}cm) { {{n.label}} };
{% endfor %}

% --- Group / lane boxes (background layer) --------------------------------
{% if boxes and boxes|length > 0 %}
\begin{scope}[on background layer]
{% for b in boxes %}
  \node[{{b.kind}}{% if b.title %},label=above:{{b.title}}{% endif %},fit={{b.tikz_fit_expr()}}] ({{b.name}}) {};
{% endfor %}
\end{scope}
{% endif %}

% --- Edges -----------------------------------------------------------------
{% for e in edges %}
  {% if e.via %}\path[{{e.style}}] {{e.src}} -- ++({{'%.2f'%e.via[0]}}cm,{{'%.2f'%e.via[1]}}cm) |- {{e.dst}};{% else %}
  \path[{{e.style}}] {{e.src}} -- {{e.dst}};{% endif %}
{% endfor %}
\end{tikzpicture}
\end{document}
"""


def export_tex(nodes: Sequence, edges: Sequence, out_tex: str | Path, boxes: Optional[Sequence] = None) -> Path:
  """Render a standalone TikZ document.

  Args:
    nodes: sequence of primitive Node objects.
    edges: sequence of Edge objects.
    out_tex: destination .tex path.
    boxes: optional sequence of Box (group / lane) objects.
  """
  out = Path(out_tex)
  out.write_text(Template(TPL).render(nodes=nodes, edges=edges, boxes=boxes or []))
  return out
