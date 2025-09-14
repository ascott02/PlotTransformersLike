"""Decoder block example (masked MHA + cross attention + FFN).
Generates examples/fig_decoder_block.tex
"""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.blocks import decoder_block_factory  # noqa: E402
from plotnn_xt.layout import repeat  # noqa: E402
from plotnn_xt.export import export_tex  # noqa: E402


def build():
    builder = decoder_block_factory(include_cross=True)
    nodes, edges, _span = builder(0, 0.0, 0.0)  # builder returns (nodes, edges, span)
    export_tex(nodes, edges, "examples/fig_decoder_block.tex")


if __name__ == "__main__":
    build()
    print("Wrote examples/fig_decoder_block.tex")
