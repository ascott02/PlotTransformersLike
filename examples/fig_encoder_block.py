"""First encoder block example using transformer extension primitives.
Generates examples/fig_encoder_block.tex
"""
import sys, pathlib

# Ensure repository root is on sys.path when running directly
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.primitives import layernorm, mha, ffn, residual_add  # noqa: E402
from plotnn_xt.export import export_tex  # noqa: E402
from plotnn_xt.layout import connect, elbow  # noqa: E402


def build():
    ln1 = layernorm("ln1", 0, 0)
    att = mha("mha", 3.2, 0)
    add1 = residual_add("add1", 5.6, 0)
    ln2 = layernorm("ln2", 7.3, 0)
    ff = ffn("ffn", 10.0, 0.2)
    add2 = residual_add("add2", 12.4, 0.2)

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
    export_tex(nodes, edges, "examples/fig_encoder_block.tex")


if __name__ == "__main__":
    build()
    print("Wrote examples/fig_encoder_block.tex")
