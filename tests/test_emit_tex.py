from plotnn_xt.primitives import layernorm, mha, ffn, residual_add
from plotnn_xt.export import export_tex
from plotnn_xt.layout import connect, elbow


def test_encoder_block_emit(tmp_path):
    ln1 = layernorm("ln1", 0, 0)
    att = mha("mha", 3.2, 0, masked=True)
    add1 = residual_add("add1", 5.6, 0)
    ln2 = layernorm("ln2", 7.3, 0)
    ff = ffn("ffn", 10.0, 0.2)
    add2 = residual_add("add2", 12.4, 0.2)

    nodes = [ln1, att, add1, ln2, ff, add2]
    edges = [
        connect(ln1.anchors["R"], att.anchors["L"]),
        elbow(ln1.anchors["L"], add1.anchors["T"], dx=-0.8),
    ]
    out = tmp_path / "enc.tex"
    export_tex(nodes, edges, out)
    text = out.read_text()
    assert "MHA*" in text  # masked label
    assert "FFN" in text
    assert "ln1" in text
    # anchor expression was rendered
    assert "(ln1.west)" in text
