from plotnn_xt.primitives import layernorm, mha, ffn, residual_add
from plotnn_xt.export import export_tex
from plotnn_xt.layout import connect, elbow, group, repeat
from plotnn_xt.blocks import encoder_block_factory, decoder_block_factory
from plotnn_xt.primitives import cls_head, patch_embed, class_token, pos_enc
from plotnn_xt.export import export_tex
from plotnn_xt.layout import group, stack_tag


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


def test_group_box_emit(tmp_path):
    # minimal two nodes + group
    ln1 = layernorm("ln1", 0, 0)
    ln2 = layernorm("ln2", 3, 0)
    g = group("g1", [ln1, ln2], title="GroupTitle")
    out = tmp_path / "grp.tex"
    export_tex([ln1, ln2], [], out, boxes=[g])
    text = out.read_text()
    # fit expression references node names
    assert "fit=(ln1) (ln2)".replace(" ", " ") in text
    # title label presence
    assert "GroupTitle" in text


def test_repeat_encoder_blocks(tmp_path):
    builder = encoder_block_factory()
    nodes, edges = repeat(2, builder, start=(0.0, 0.0), gap=1.5, dir="x")
    out = tmp_path / "stack.tex"
    export_tex(nodes, edges, out)
    text = out.read_text()
    # Expect two MHA nodes and two FFN nodes
    assert text.count("MHA") >= 2
    assert text.count("FFN") >= 2


def test_decoder_block_emit(tmp_path):
    builder = decoder_block_factory(include_cross=True)
    nodes, edges, _ = builder(0, 0.0, 0.0)
    out = tmp_path / "decoder.tex"
    export_tex(nodes, edges, out)
    text = out.read_text()
    assert "MHA*" in text  # masked self-attention
    assert "CrossAttn" in text


def test_vit_patchflow_min(tmp_path):
    # Mini reproduction (1 encoder block) of ViT flow focusing on CLS head emission
    pe = pos_enc("pos", 0.0, 0.0)
    head = cls_head("head", 4.0, 0.0, n_classes=10)
    nodes = [pe, head]
    out = tmp_path / "vitmini.tex"
    export_tex(nodes, [], out)
    txt = out.read_text()
    assert "CLS Head" in txt
    assert "$C=10$" in txt

    def test_legacy_transformer_block(tmp_path):
        """Ensure legacy macro emits expected labels (MHA, FFN, CLS Head optional)."""
        # build a tiny legacy arch manually
        from pycore import tikzeng as T  # type: ignore
        arch = [T.to_head('..'), T.to_begin()]
        arch += T.to_transformer_block("t0_", 0.0, 0.0)
        arch.append(T.to_end())
        out = tmp_path / "legacy_test.tex"
        T.to_generate(arch, str(out))
        data = out.read_text()
        assert "MHA" in data
        assert "FFN" in data

