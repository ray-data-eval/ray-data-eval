from __future__ import annotations

import shutil
import warnings

import matplotlib
import matplotlib.pyplot as plt

FIGWIDTH = 3.335
BIG_SIZE = 14

HAVE_LATEX = shutil.which("latex") is not None


def use_style(
    figratio: float = 3 / 4,
    size: int = BIG_SIZE,
    palette: str = "Paired",
    width: float = FIGWIDTH,
):
    """Apply the paper's matplotlib settings.

    Returns the colour list to draw with.
    """
    try:
        import seaborn as sns

        colors = sns.color_palette(palette)
        sns.set_style("ticks")
        sns.set_palette(colors)
    except ImportError:
        warnings.warn(
            "seaborn is not installed; falling back to matplotlib tab10. "
            "To install: `pip install seaborn`",
            stacklevel=2,
        )
        colors = [plt.get_cmap("tab10")(i) for i in range(10)]

    plt.rcParams.update(
        {
            "figure.figsize": (width, width * figratio),
            "figure.dpi": 300,
            "text.usetex": HAVE_LATEX,
        }
    )
    for key in ("font", "axes", "xtick", "ytick", "legend", "figure"):
        plt.rc(key, **({"size": size} if key == "font" else {}))
    plt.rc("axes", titlesize=size, labelsize=size)
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)
    plt.rc("legend", fontsize=size)
    plt.rc("figure", titlesize=size)
    return colors


def tex(s: str) -> str:
    """Make a label safe when LaTeX rendering is off."""
    if HAVE_LATEX:
        return s
    return s.replace("\\times", "x").replace("$", "").replace("\\", "")


def save(fig_or_plt, outdir: str, name: str):
    """Write the current figure as both PDF and PNG."""
    import os

    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig_or_plt.savefig(path, bbox_inches="tight")
        written.append(path)
    print(f"  wrote {written[0]}")
    print(f"  wrote {written[1]}")
    return written


matplotlib.use("Agg")
