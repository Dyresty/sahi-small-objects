"""Plot accuracy vs latency from evaluate.py results (two panels: AP and AP_small).

Usage:
    python scripts/plot.py outputs/eval/results_val.csv
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e8e7e3", "#fcfcfb"
SERIES = {"full image": "#2a78d6", "SAHI": "#eb6834", "Ultralytics tile": "#1baf7a"}  # validated palette
MARKERS = {"full image": "o", "SAHI": "s", "Ultralytics tile": "^"}  # shape as 2nd encoding
LABEL_OFFSET = {"SAHI": (6, 6), "Ultralytics tile": (6, -14)}  # SAHI labels above, tile below: avoid collisions


def group(mode: str) -> tuple[str, str]:
    kind, _, size = mode.partition(":")
    return {"standard": "full image", "sahi": "SAHI", "tile": "Ultralytics tile"}.get(kind, ""), size


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval/results_val.csv")
    rows = list(csv.DictReader(open(src)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, metric, title in zip(axes, ["AP", "AP_small"], ["mAP50-95 (all objects)", "mAP50-95 (small objects, < 32² px)"]):
        ax.set_facecolor(SURFACE)
        for name, color in SERIES.items():
            pts = sorted(((float(r["ms_per_img"]), float(r[metric]), size)
                         for r in rows for g, size in [group(r["mode"])] if g == name),
                         key=lambda p: int(p[2] or 0))  # line follows the slice-size sweep
            if not pts:
                continue
            xs, ys, sizes = zip(*pts)
            ax.plot(xs, ys, color=color, lw=2, marker=MARKERS[name], ms=8, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            for x, y, s in zip(xs, ys, sizes):
                if s:
                    ax.annotate(f"{s}px", (x, y), textcoords="offset points", xytext=LABEL_OFFSET[name], fontsize=8, color=MUTED)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([25, 50, 100, 200, 400, 800, 1600]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("latency per image, ms (log scale)", color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("VisDrone val: accuracy vs speed (labels = slice / tile size)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    out = src.with_suffix(".png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
