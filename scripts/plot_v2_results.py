"""Figure of the V2 test results: track macro-F1 with BCa 95 % intervals, from the archived report.

  python scripts/plot_v2_results.py

Writes docs/figures/test_v2_macro_f1.png and .svg. Reads only reports/v2/final_evaluation/report.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = [  # (report key, label), drawn top to bottom
    ("maest_layer12", "MAEST block 12, frozen + linear probe (V2)"),
    ("vit_block6_hpss", "Image ViT block 6 + HPSS (V1 candidate)"),
    ("vit_block6", "Image ViT block 6"),
    ("vit", "Image ViT block 12"),
    ("cnn_v2_seed44", "Reinforced CNN, seed 44"),
    ("cnn_v2_seed42", "Reinforced CNN, seed 42"),
    ("cnn_v2_seed43", "Reinforced CNN, seed 43"),
]
ACCENT, CONTEXT, MUTED, GRID, INK, SURFACE = "#2a78d6", "#52514e", "#898781", "#e1e0d9", "#0b0b0b", "#fcfcfb"


def main():
    report = json.loads((ROOT / "reports/v2/final_evaluation/report.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for row, (key, label) in enumerate(SYSTEMS):
        f1 = report["systems"][key]["macro_f1"]
        color = ACCENT if row == 0 else CONTEXT
        y = len(SYSTEMS) - 1 - row
        ax.plot([f1["low"], f1["high"]], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(f1["point"], y, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(f1["high"] + 0.006, y, f"{f1['point']:.3f}", va="center", fontsize=9,
                color=INK, fontweight="bold" if row == 0 else "normal")
    ax.set_yticks(range(len(SYSTEMS)), [label for _, label in reversed(SYSTEMS)], fontsize=9, color=INK)
    ax.set_xlim(0.36, 0.66)
    ax.set_xlabel("Track macro-F1 on the V2 test (501 tracks, one per artist), BCa 95 % interval",
                  fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout()
    out = ROOT / "docs/figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"test_v2_macro_f1.{suffix}", facecolor=SURFACE, metadata={"Date": None} if suffix == "svg" else None)
    print(out / "test_v2_macro_f1.png")


if __name__ == "__main__":
    main()
