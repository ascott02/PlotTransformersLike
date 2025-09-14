"""Export utilities (Jinja -> standalone TikZ)."""
from __future__ import annotations
from pathlib import Path
from jinja2 import Template
from typing import Sequence

TPL = r"""
\documentclass[tikz,border=2pt]{standalone}
\input{transformer_tex/transformer_styles.tex}
\begin{document}
\begin{tikzpicture}
{% for n in nodes %}
  \node[{{n.kind}}={{'%.2f'%n.w}}cm/{{'%.2f'%n.h}}cm] ({{n.name}}) at ({{'%.2f'%n.x}}cm,{{'%.2f'%n.y}}cm) { {{n.label}} };
{% endfor %}
{% for e in edges %}
  {% if e.via %}\path[{{e.style}}] {{e.src}} -- ++({{'%.2f'%e.via[0]}}cm,{{'%.2f'%e.via[1]}}cm) |- {{e.dst}};{% else %}
  \path[{{e.style}}] {{e.src}} -- {{e.dst}};{% endif %}
{% endfor %}
\end{tikzpicture}
\end{document}
"""


def export_tex(nodes: Sequence, edges: Sequence, out_tex: str | Path) -> Path:
    out = Path(out_tex)
    out.write_text(Template(TPL).render(nodes=nodes, edges=edges))
    return out
