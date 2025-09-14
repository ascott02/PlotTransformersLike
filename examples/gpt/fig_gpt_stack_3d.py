"""3D-stylized GPT decoder stack example (uses three_d flag on factory)."""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotnn_xt.gpt import build_gpt_stack  # type: ignore  # noqa: E402


def build(n: int = 3):
    out = _ROOT / "examples" / "gpt" / "fig_gpt_stack_3d.tex"
    build_gpt_stack(n=n, three_d=True, include_cross=False, out_tex=str(out))
    return str(out)


if __name__ == "__main__":  # pragma: no cover
    p = build(4)
    print("Wrote", p)
