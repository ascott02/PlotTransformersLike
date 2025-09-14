"""Encoder stack example using repeat utility.
Generates examples/fig_encoder_stack.tex
"""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.blocks import encoder_block_factory  # noqa: E402
from plotnn_xt.layout import repeat  # noqa: E402
from plotnn_xt.export import export_tex  # noqa: E402


def build(n: int = 3):
    builder = encoder_block_factory()
    nodes, edges = repeat(n, builder, start=(0.0, 0.0), gap=2.0, dir="y")  # vertical stack
    export_tex(nodes, edges, f"examples/fig_encoder_stack.tex")


if __name__ == "__main__":
    build()
    print("Wrote examples/fig_encoder_stack.tex")
