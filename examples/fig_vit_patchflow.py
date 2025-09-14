"""Vision Transformer patch flow example (stub).
PatchEmbed -> [CLS] + PosEnc concat (conceptually) -> single encoder block (placeholder) -> CLS head placeholder.
Generates examples/fig_vit_patchflow.tex
"""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.primitives import patch_embed, class_token, pos_enc, layernorm, cls_head  # type: ignore  # noqa: E402
from plotnn_xt.blocks import encoder_block_factory  # type: ignore  # noqa: E402
from plotnn_xt.layout import connect, elbow, group, repeat, stack_tag  # type: ignore  # noqa: E402
from plotnn_xt.export import export_tex  # type: ignore  # noqa: E402


def build(n_enc: int = 4):
    """Build a fuller ViT patch flow with N encoder blocks and a CLS head.

    n_enc: number of encoder blocks to depict.
    """
    # Base primitives
    patch = patch_embed("patch", 0.0, 0.0)
    cls = class_token("cls", patch.x + patch.w + 1.2, 0.2)
    pe = pos_enc("pos", cls.x + cls.w + 0.9, 0.2)

    enc_builder = encoder_block_factory()
    # horizontally place first encoder block start after pos enc with a gap
    first_x = pe.x + pe.w + 1.6
    enc_nodes_all = []
    enc_edges_all = []
    # Use repeat along x axis manually to preserve existing builder span semantics
    ox = first_x
    for i in range(n_enc):
        nodes_i, edges_i, span = enc_builder(i, ox, 0.0)
        enc_nodes_all.extend(nodes_i)
        enc_edges_all.extend(edges_i)
        ox += span + 2.0  # gap between encoder blocks

    # Classification head placed after final encoder output
    head = cls_head("cls_head", ox, 0.1, n_classes=1000)

    nodes = [patch, cls, pe] + enc_nodes_all + [head]
    # base flow edges
    edges = [
        connect(patch.anchors["R"], cls.anchors["L"]),
        connect(cls.anchors["R"], pe.anchors["L"]),
        elbow(patch.anchors["T"], pe.anchors["T"], dx=0.4),  # conceptual concat
        # connect last encoder residual add to head
        connect(enc_nodes_all[-1].anchors["R"], head.anchors["L"]),
    ]
    # also include the first edge of each encoder block for minimal depiction
    for i in range(n_enc):
        # each block's first 6 nodes; first edge index 0 of slice
        block_nodes = enc_nodes_all[i * 6 : (i + 1) * 6]
        block_edges = enc_edges_all[i * 7 : (i + 1) * 7]  # 7 edges per encoder block currently
        if block_edges:
            edges.append(block_edges[0])

    g_stack = group("enc_stack", enc_nodes_all, title=f"Encoder ×{n_enc}")
    tag = stack_tag("enc_tag", enc_nodes_all, text=f"$\\times {n_enc}$")
    export_tex(nodes + [tag], edges, "examples/fig_vit_patchflow.tex", boxes=[g_stack])
    return "examples/fig_vit_patchflow.tex"


if __name__ == "__main__":  # pragma: no cover
    print("Wrote", build())
