"""GPT-style decoder stack example (flat 2D)."""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.gpt import build_gpt_stack  # type: ignore  # noqa: E402


def build(n: int = 3):
    out = _ROOT / "examples" / "gpt" / "fig_gpt_stack.tex"
    build_gpt_stack(n=n, three_d=False, include_cross=False, out_tex=str(out))
    return str(out)


if __name__ == "__main__":  # pragma: no cover
    p = build(4)
    print("Wrote", p)
