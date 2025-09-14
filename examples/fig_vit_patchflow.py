"""Vision Transformer patch flow example (stub).
PatchEmbed -> [CLS] + PosEnc concat (conceptually) -> single encoder block (placeholder) -> CLS head placeholder.
Generates examples/fig_vit_patchflow.tex
"""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.primitives import patch_embed, class_token, pos_enc, layernorm  # type: ignore  # noqa: E402
from plotnn_xt.blocks import encoder_block_factory  # type: ignore  # noqa: E402
from plotnn_xt.layout import connect, elbow, group  # type: ignore  # noqa: E402
from plotnn_xt.export import export_tex  # type: ignore  # noqa: E402


def build():
    # Base primitives
    patch = patch_embed("patch", 0.0, 0.0)
    cls = class_token("cls", patch.x + patch.w + 1.2, 0.2)
    pe = pos_enc("pos", cls.x + cls.w + 0.9, 0.2)

    # Single encoder block placed after positional encoding (simplified)
    enc_builder = encoder_block_factory()
    enc_nodes, enc_edges, span = enc_builder(0, pe.x + pe.w + 1.6, 0.0)

    nodes = [patch, cls, pe] + enc_nodes
    edges = [
        connect(patch.anchors["R"], cls.anchors["L"]),
        connect(cls.anchors["R"], pe.anchors["L"]),
        # conceptually a concat: we indicate with an elbow from patch top to pe top
        elbow(patch.anchors["T"], pe.anchors["T"], dx=0.4),
    ] + enc_edges[:1]  # only first edge of encoder for brevity

    g_enc = group("enc_block", enc_nodes, title="Encoder ×1 (stub)")
    export_tex(nodes, edges, "examples/fig_vit_patchflow.tex", boxes=[g_enc])
    return "examples/fig_vit_patchflow.tex"


if __name__ == "__main__":  # pragma: no cover
    print("Wrote", build())
