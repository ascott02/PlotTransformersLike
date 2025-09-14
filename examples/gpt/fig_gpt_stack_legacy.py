"""Legacy GPT stack example using pycore.tikzeng compatibility layer (2D)."""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pycore import tikzeng as T  # type: ignore  # noqa: E402


def build(n: int = 3):
    arch = [T.to_head('..'), T.to_begin()]
    for i in range(n):
        arch += T.to_transformer_block(f"t{i}_", i * 12.5, 0.0)
    arch.append(T.to_end())
    out = _ROOT / "examples" / "gpt" / "fig_gpt_stack_legacy.tex"
    T.to_generate(arch, str(out))
    return str(out)


if __name__ == "__main__":  # pragma: no cover
    p = build(3)
    print("Wrote", p)
