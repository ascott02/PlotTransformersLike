"""Legacy-style GPT stack example using tikzeng to_generate with transformer compat snippets.
Generates examples/fig_gpt_stack_legacy.tex
"""
import sys, pathlib, os
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    # also ensure current working directory is repository root so relative style path works
    try:
        os.chdir(str(_ROOT))  # type: ignore
    except Exception:
        pass

from pycore import tikzeng  # type: ignore


def build(n_blocks: int = 3):
    arch = []
    project_path = str(_ROOT)
    arch.append(tikzeng.to_head(project_path))
    arch.append(tikzeng.to_begin())

    x0 = 0.0
    y0 = 0.0
    # stack of minimal transformer encoder-like blocks representing causal stack
    for i in range(n_blocks):
        arch.extend(tikzeng.to_xt_transformer_block(f"b{i}_", x0, y0))
        x0 += 18.0  # coarse spacing per micro-block

    arch.append(tikzeng.to_end())
    out = "examples/fig_gpt_stack_legacy.tex"
    tikzeng.to_generate(arch, out)
    return out


if __name__ == "__main__":  # pragma: no cover
    print("Wrote", build())
