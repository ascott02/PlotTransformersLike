"""GPT-style decoder stack example.
Builds N repeated decoder blocks WITHOUT cross-attention (pure causal decoder stack) and exports examples/fig_gpt_stack.tex
"""
import sys, pathlib
from typing import Tuple, List

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.blocks import decoder_block_factory  # type: ignore  # noqa: E402
from plotnn_xt.layout import repeat, group, stack_tag  # type: ignore  # noqa: E402
from plotnn_xt.export import export_tex  # type: ignore  # noqa: E402


def build(n: int = 3) -> Tuple[str, List]:
    builder = decoder_block_factory(include_cross=False)
    nodes, edges = repeat(n, builder, start=(0.0, 0.0), gap=2.0, dir="x")
    # simple group box around entire stack
    g = group("gpt_stack", nodes, title=f"Decoder ×{n}")
    tag = stack_tag("gpt_tag", nodes, text=f"$\\times {n}$")
    export_tex(nodes + [tag], edges, "examples/fig_gpt_stack.tex", boxes=[g])
    return "examples/fig_gpt_stack.tex", nodes


if __name__ == "__main__":  # pragma: no cover
    path, _ = build(4)
    print("Wrote", path)
