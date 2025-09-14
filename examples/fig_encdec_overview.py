"""Encoder–Decoder overview figure.
Builds N encoder blocks (vertical) feeding into M decoder blocks (vertical) with cross-attention edges from final encoder outputs to each decoder cross-attn node.
Generates examples/fig_encdec_overview.tex
"""
import sys, pathlib
from typing import List, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.blocks import encoder_block_factory, decoder_block_factory  # type: ignore  # noqa: E402
from plotnn_xt.layout import repeat, group, connect, elbow  # type: ignore  # noqa: E402
from plotnn_xt.export import export_tex  # type: ignore  # noqa: E402


def build(n_enc: int = 3, n_dec: int = 2) -> str:
    enc_builder = encoder_block_factory()
    dec_builder = decoder_block_factory(include_cross=True)

    # Encoder stack vertical at origin
    enc_nodes, enc_edges = repeat(n_enc, enc_builder, start=(0.0, 0.0), gap=1.6, dir="y")
    g_enc = group("enc_stack", enc_nodes, title=f"Encoder ×{n_enc}")

    # Determine rightmost x of encoder stack to start decoder
    enc_right = max(n.x + getattr(n, "w", 0) for n in enc_nodes)
    dec_start_x = enc_right + 6.0
    dec_nodes, dec_edges = repeat(n_dec, dec_builder, start=(dec_start_x, 0.0), gap=2.0, dir="y")
    g_dec = group("dec_stack", dec_nodes, title=f"Decoder ×{n_dec}")

    # Cross-attention nodes inside decoder blocks have names xatt{idx}
    # Build edges from final encoder residual add node (use last node) to each xatt
    cross_edges: List = []
    enc_out = enc_nodes[-1]
    for dn in dec_nodes:
        if dn.name.startswith("xatt"):
            cross_edges.append(elbow(enc_out.anchors["R"], dn.anchors["L"], dx=2.0))

    nodes = enc_nodes + dec_nodes
    edges = enc_edges + dec_edges + cross_edges
    export_tex(nodes, edges, "examples/fig_encdec_overview.tex", boxes=[g_enc, g_dec])
    return "examples/fig_encdec_overview.tex"


if __name__ == "__main__":  # pragma: no cover
    print("Wrote", build())
