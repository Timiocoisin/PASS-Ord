#!/usr/bin/env python3
"""Generate Sensors/top-venue style paper figures from closed experimental tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs/baselines/paper_tables"
OUT = ROOT / "outputs/baselines/paper_figures"
PAN = ROOT / "outputs/processed/panoramas"
SEG = ROOT / "outputs/processed/segmentation/overlays"
SEG_CSV = ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv"

# Okabe–Ito + neutrals (colorblind-safe, Nature/CVPR-like)
C = {
    "black": "#000000",
    "gray": "#4D4D4D",
    "light": "#B0B0B0",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "ours": "#0072B2",
    "base": "#7A7A7A",
}


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save(fig, name: str, pad_inches: float = 0.06):
    OUT.mkdir(parents=True, exist_ok=True)
    # Explicit DPI: PNG must be print-sharp when stretched to ~0.95\textwidth;
    # PDF remains vector (matplotlib paths/text). Do not rely on rc alone.
    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    fig.savefig(
        OUT / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    plt.close(fig)
    print(f"  wrote {name}.pdf/png", flush=True)


def box(ax, xy, w, h, text, fc="#F7F7F7", ec=C["black"], lw=0.9, fontsize=7.5, weight="normal"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=C["black"],
        wrap=True,
    )
    return p


def arrow(ax, p1, p2, color=C["gray"]):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=9),
    )


def fig01_overview():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) task / protocol
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(a) Problem & label protocol", loc="left", fontweight="bold", pad=6)
    box(ax, (0.3, 7.2), 4.2, 2.0, "Boiler water-wall\ngrinding sequence\n(g = 1…169)", fc="#EEF6FB", ec=C["blue"])
    box(ax, (5.3, 7.4), 4.2, 1.6, "Pass-level vision\n(~5 passes / grind)", fc="#F7F7F7")
    arrow(ax, (4.5, 8.2), (5.3, 8.2))
    box(ax, (0.3, 4.4), 2.9, 1.8, "cls3\nearly/mid/late", fc="#FFF6E8", ec=C["orange"])
    box(ax, (3.5, 4.4), 2.9, 1.8, "cls5\n5 ordinal bins", fc="#FFF6E8", ec=C["orange"])
    box(ax, (6.7, 4.4), 2.9, 1.8, "progress\ng/169 ∈ [0,1]", fc="#EAF7F2", ec=C["green"])
    box(
        ax,
        (0.3, 1.05),
        9.3,
        2.45,
        "Ranking: nested even/odd-stage hold-out\n"
        "odd → test · even $g\\equiv0\\pmod{4}$ → fit · $g\\equiv2\\pmod{4}$ → val\n"
        "Stress only (not ranked): spatial hold-out · random frame split",
        fc="#F7F7F7",
        fontsize=6.8,
    )
    ax.text(0.3, 0.35, "No physical thickness GT · operation-order (index) supervision", fontsize=7, color=C["gray"])

    # (b) qualitative progression
    ax = axes[1]
    ax.set_title("(b) Appearance along grinding", loc="left", fontweight="bold", pad=6)
    picks = [
        ("grind_001_pass0.png", "g=1"),
        ("grind_050_pass2.png", "g=50"),
        ("grind_100_pass0.png", "g=100"),
        ("grind_169_pass0.png", "g=169"),
    ]
    for i, (fn, lab) in enumerate(picks):
        img = np.asarray(Image.open(PAN / fn).convert("RGB"))
        # crop center band for readability
        h, w = img.shape[:2]
        y0, y1 = int(0.15 * h), int(0.85 * h)
        img = img[y0:y1]
        ax_i = ax.inset_axes([0.02, 0.72 - i * 0.23, 0.96, 0.20])
        ax_i.imshow(img, aspect="auto")
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        for sp in ax_i.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color(C["light"])
        ax_i.text(
            0.01,
            0.55,
            lab,
            transform=ax_i.transAxes,
            color="white",
            fontsize=7,
            fontweight="bold",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc=C["black"], ec="none", alpha=0.55),
        )
    ax.axis("off")
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig01_overview")


def _layer_stack(ax, x, y, w, h, n, fc, ec, label=None, gap=0.06):
    """Draw a 2D stack of layer cards (CVPR-style block)."""
    for i in range(n):
        yy = y + i * gap
        ax.add_patch(
            FancyBboxPatch(
                (x + i * 0.04, yy),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                linewidth=0.8,
                edgecolor=ec,
                facecolor=fc,
                zorder=2 + i,
            )
        )
    if label:
        ax.text(x + w / 2 + (n - 1) * 0.02, y + h / 2 + (n - 1) * gap / 2, label,
                ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=10, color=C["black"])
    return x + w + (n - 1) * 0.04, y + h + (n - 1) * gap


def _conn(ax, x1, y1, x2, y2, color=C["gray"], lw=1.0, rad=0.0):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=8,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=1,
    )


def fig02_method():
    """Minimal-text Nature/CVPR-style PASS-Ord diagram (pastel modules, feature bars)."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    # soft pastels (reference-like)
    COL = {
        "ink": "#2B2B2B",
        "mute": "#6B6B6B",
        "line": "#3A3A3A",
        "panel": "#F2F2F2",
        "blue_bg": "#DCEAF6",
        "blue": "#4A90C8",
        "peach_bg": "#F7E4D4",
        "peach": "#E09A5B",
        "green_bg": "#D9EFE3",
        "green": "#3D9B78",
        "lav_bg": "#E8DFF0",
        "lav": "#8E6BA8",
        "cream": "#FFF7EF",
        "white": "#FFFFFF",
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.35))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    def panel(x, y, w, h, fc, ec, lw=1.3, rs=0.12):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0.012,rounding_size={rs}",
                lw=lw, ec=ec, fc=fc, zorder=1,
            )
        )

    def tag(x, y, s, color=COL["ink"]):
        ax.text(x, y, s, ha="left", va="center", fontsize=8.5, fontweight="bold",
                color=color, zorder=6)

    def label(x, y, s, fs=7.0, color=COL["ink"], weight="bold", ha="center", va="center"):
        ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=weight, color=color, zorder=6)

    def feat_bar(x, y, colors, bw=0.28, bh=0.95, gap=0.04):
        """Vertical segmented feature strip."""
        for i, c in enumerate(colors):
            ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, bh, fc=c, ec="none", zorder=5))

    def arr(x1, y1, x2, y2, color=COL["line"], lw=1.15, rad=0.0):
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, mutation_scale=10,
                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=1,
            ),
            zorder=4,
        )

    def op_circle(x, y, sym, fc=COL["white"], ec=COL["line"]):
        ax.add_patch(Circle((x, y), 0.22, fc=fc, ec=ec, lw=1.1, zorder=7))
        ax.text(x, y, sym, ha="center", va="center", fontsize=8, fontweight="bold",
                color=COL["ink"], zorder=8)

    # ========== (a) Input ==========
    panel(0.25, 1.1, 2.55, 7.5, COL["panel"], "#B8B8B8", lw=1.1)
    tag(0.4, 8.25, "(a) Input")

    picks = [
        ("grind_001_pass0.png", "early"),
        ("grind_050_pass2.png", "mid"),
        ("grind_169_pass0.png", "late"),
    ]
    for i, (fn, lab) in enumerate(picks):
        path = PAN / fn
        if not path.exists():
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        img = img[int(0.18 * h):int(0.82 * h)]
        y0 = 6.35 - i * 2.35
        panel(0.45, y0, 2.15, 2.05, COL["white"], "#C8C8C8", lw=0.8, rs=0.06)
        ax.imshow(img, extent=[0.55, 2.45, y0 + 0.35, y0 + 1.85],
                  aspect="auto", zorder=3, interpolation="bilinear")
        label(1.5, y0 + 0.18, lab, fs=6.2, color=COL["mute"], weight="normal")

    # ========== (b) Experts ==========
    panel(3.15, 4.55, 4.7, 4.05, COL["blue_bg"], COL["blue"], lw=1.25)
    tag(3.3, 8.25, "(b) Experts", COL["blue"])

    # three expert mini-cards
    experts = [
        (3.4, 6.85, COL["blue"], "#EAF3FA", "Temporal", "BiLSTM"),
        (3.4, 5.55, COL["peach"], "#FBF0E4", "Statistical", "RF"),
        (3.4, 4.75, COL["green"], "#E8F6EF", "Appearance", "ResNet"),
    ]
    # redraw as three horizontal rows inside (b)
    rows = [
        (6.95, COL["blue"], "#EAF3FA", "Temporal", "BiLSTM", [COL["blue"], "#7EB3D9", "#A8CDE6"]),
        (5.75, COL["peach"], "#FBF0E4", "Statistical", "Pool+RF", [COL["peach"], "#E8B87A", "#F0D0A8"]),
        (4.75, COL["green"], "#E8F6EF", "Appearance", "ResNet-18", [COL["green"], "#6FBF9A", "#A5D9C0"]),
    ]
    for y, ec, fc, title, sub, bars in rows:
        panel(3.4, y, 4.25, 1.05, fc, ec, lw=1.0, rs=0.08)
        label(3.65, y + 0.55, title, fs=7.2, color=ec, ha="left")
        label(5.55, y + 0.55, sub, fs=6.2, color=COL["mute"], weight="normal", ha="left")
        feat_bar(6.55, y + 0.12, bars, bw=0.26, bh=0.8)

    # input → experts
    arr(2.8, 5.4, 3.15, 6.5, COL["line"], rad=-0.05)
    arr(2.8, 4.0, 3.15, 5.3, COL["line"])
    arr(2.8, 2.5, 3.15, 5.0, COL["line"], rad=0.08)

    # ========== (c) Soft task routing (CORE) ==========
    panel(3.15, 0.35, 8.35, 3.9, COL["cream"], COL["peach"], lw=1.45)
    tag(3.3, 3.9, "(c) Soft task routing", COL["peach"])

    # left: expert outputs as bars
    for i, (name, cols, y) in enumerate([
        ("T", [COL["blue"], "#7EB3D9"], 2.55),
        ("S", [COL["peach"], "#E8B87A"], 1.55),
        ("A", [COL["green"], "#6FBF9A"], 0.55),
    ]):
        label(3.55, y + 0.45, name, fs=7.5, color=COL["ink"], ha="left")
        feat_bar(4.0, y + 0.1, cols, bw=0.32, bh=0.7)

    # mix ops
    op_circle(5.55, 2.35, r"$\oplus$")
    op_circle(5.55, 1.05, r"$\oplus$")
    arr(4.75, 2.7, 5.3, 2.45, COL["blue"], lw=1.0)
    arr(4.75, 1.85, 5.3, 2.25, COL["peach"], lw=1.0)
    arr(4.75, 1.7, 5.3, 1.15, COL["peach"], lw=1.0)
    arr(4.75, 0.85, 5.3, 0.95, COL["green"], lw=1.0)
    # weak T → progress mix
    ax.annotate(
        "", xy=(5.3, 0.85), xytext=(4.75, 2.55),
        arrowprops=dict(arrowstyle="-|>", color=COL["blue"], lw=0.8, mutation_scale=8,
                        linestyle=(0, (2, 1.5)), connectionstyle="arc3,rad=0.18"),
        zorder=4,
    )

    # weight chips (minimal numbers)
    panel(6.0, 2.0, 2.35, 1.15, COL["white"], COL["blue"], lw=1.0, rs=0.07)
    label(7.175, 2.75, "cls route", fs=6.5, color=COL["blue"])
    label(7.175, 2.25, r"$0.90{:}0.10$", fs=7.5, color=COL["ink"])
    label(7.175, 1.85, r"$0.25{:}0.75$", fs=7.5, color=COL["ink"])

    panel(6.0, 0.55, 2.35, 1.25, COL["white"], COL["green"], lw=1.0, rs=0.07)
    label(7.175, 1.45, "prog fuse", fs=6.5, color=COL["green"])
    label(7.175, 0.95, r"$0.68{:}0.25{:}0.07$", fs=7.2, color=COL["ink"])

    arr(5.8, 2.35, 6.0, 2.55, COL["line"], lw=1.0)
    arr(5.8, 1.05, 6.0, 1.15, COL["line"], lw=1.0)

    # fused bars out
    feat_bar(8.7, 2.15, [COL["blue"], COL["peach"]], bw=0.3, bh=0.85)
    feat_bar(8.7, 0.65, [COL["green"], COL["peach"], COL["blue"]], bw=0.28, bh=0.85)
    arr(8.35, 2.55, 8.7, 2.55, COL["blue"], lw=0.95)
    arr(8.35, 1.15, 8.7, 1.05, COL["green"], lw=0.95)

    label(9.85, 2.55, r"$\tilde p$", fs=8, color=COL["blue"], ha="left")
    label(9.85, 1.05, r"$\hat y$", fs=8, color=COL["green"], ha="left")

    # experts → routing
    arr(5.5, 4.55, 5.5, 3.95, COL["line"], lw=1.1)

    # ========== (d) Heads ==========
    panel(11.85, 4.55, 4.05, 4.05, COL["lav_bg"], COL["lav"], lw=1.25)
    tag(12.0, 8.25, "(d) Heads", COL["lav"])

    coords = [(12.15, 6.55), (14.0, 6.55), (12.15, 4.85), (14.0, 4.85)]
    styles = [
        (COL["blue"], "#EAF3FA", "cls3"),
        (COL["peach"], "#FBF0E4", "cls5"),
        (COL["green"], "#E8F6EF", "progress"),
        (COL["lav"], "#F3EAF8", "plateau"),
    ]
    for (x, y), (ec, fc, name) in zip(coords, styles):
        panel(x, y, 1.65, 1.35, fc, ec, lw=1.05, rs=0.08)
        label(x + 0.825, y + 0.7, name, fs=7.5, color=ec)

    # clean rightward then upward stubs into heads
    ax.plot([10.4, 11.35, 11.35], [2.55, 2.55, 7.2], color=COL["blue"], lw=1.05, zorder=3)
    arr(11.35, 7.2, 12.15, 7.2, COL["blue"], lw=1.05)
    ax.plot([10.4, 11.55, 11.55], [2.35, 2.35, 7.2], color=COL["peach"], lw=1.0, zorder=3)
    arr(11.55, 7.2, 14.0, 7.2, COL["peach"], lw=1.0)
    ax.plot([10.4, 11.35, 11.35], [1.05, 1.05, 5.5], color=COL["green"], lw=1.05, zorder=3)
    arr(11.35, 5.5, 12.15, 5.5, COL["green"], lw=1.05)

    # ========== (e) Output ==========
    panel(11.85, 0.35, 4.05, 3.9, COL["green_bg"], COL["green"], lw=1.25)
    tag(12.0, 3.9, "(e) Output", COL["green"])

    ax_in = fig.add_axes([0.755, 0.105, 0.205, 0.28])
    pts = OUT / "_progress_points.npz"
    meta_csv = ROOT / "outputs/baselines/features/frame_features_meta.csv"
    if pts.exists() and meta_csv.exists():
        d = np.load(pts)
        meta = pd.read_csv(meta_csv)
        te = meta[meta["split_interleaved"] == "test"].reset_index(drop=True)
        n = min(len(te), len(d["yt"]))
        te = te.iloc[:n].copy()
        te["yt"], te["yp"] = d["yt"][:n], d["yp"][:n]
        gg = te.groupby("grind", as_index=False).agg(yt=("yt", "mean"), yp=("yp", "mean"))
        g, yt, yp = gg["grind"].to_numpy(), gg["yt"].to_numpy(), gg["yp"].to_numpy()
    else:
        g = np.arange(1, 170)
        yt = yp = g / 169.0
    ax_in.axvspan(1, 40.5, color=COL["blue"], alpha=0.12, lw=0)
    ax_in.axvspan(40.5, 110.5, color=COL["peach"], alpha=0.12, lw=0)
    ax_in.axvspan(110.5, 169, color=COL["green"], alpha=0.12, lw=0)
    ax_in.plot(g, yt, color="#B0B0B0", lw=1.2)
    ax_in.plot(g, yp, color=COL["green"], lw=1.5)
    ax_in.set_xlim(1, 169)
    ax_in.set_ylim(0, 1.02)
    ax_in.set_xticks([])
    ax_in.set_yticks([0, 1])
    ax_in.tick_params(labelsize=5.5, length=1.5, colors=COL["mute"])
    for sp in ax_in.spines.values():
        sp.set_color("#C8C8C8")
        sp.set_linewidth(0.7)
    ax_in.set_ylabel("prog.", fontsize=6, color=COL["mute"])

    save(fig, "Fig02_method_PASSOrd")



def fig03_main():
    """Horizontal dot plot: avoids truncated-axis exaggeration of bar charts."""
    # Prefer English closed table; fall back to Chinese export if needed.
    nested = TAB / "T4_nested_main.csv"
    if nested.exists():
        raw = pd.read_csv(nested)
        order_keys = [
            "B1_SVM",
            "B2_RF",
            "B3_ResNet18",
            "B4_EfficientNetB0",
            "B5_MeanPool_RF",
            "B6_LSTM",
            "Ours_PASSOrd",
        ]
        raw["model"] = pd.Categorical(raw["model"], categories=order_keys, ordered=True)
        raw = raw.dropna(subset=["model"]).sort_values("model")
        labels = ["B1", "B2", "B3", "B4", "B5", "B6", "Ours"]
        vals = {
            "acc3": raw["cls3_acc"].to_numpy(float),
            "acc5": raw["cls5_acc"].to_numpy(float),
            "mae": raw["progress_mae"].to_numpy(float),
            "rho": raw["progress_spearman"].to_numpy(float),
        }
    else:
        raw = pd.read_csv(TAB / "T4_main_comparison.csv")
        raw = raw[raw["split"] == "interleaved"].copy()
        order_keys = [
            "B1_SVM",
            "B2_RF",
            "B3_ResNet18",
            "B4_EfficientNetB0",
            "B5_MeanPool_RF",
            "B6_LSTM",
            "Ours_PASSOrd_v9",
        ]
        raw["model"] = pd.Categorical(raw["model"], categories=order_keys, ordered=True)
        raw = raw.sort_values("model")
        labels = ["B1", "B2", "B3", "B4", "B5", "B6", "Ours"]
        vals = {
            "acc3": raw["cls3_acc"].to_numpy(float),
            "acc5": raw["cls5_acc"].to_numpy(float),
            "mae": raw["progress_mae"].to_numpy(float),
            "rho": raw["progress_spearman"].to_numpy(float),
        }

    # Framework-figure blue (Fig. 2 temporal expert)
    ours_c = "#4A90C8"
    base_c = "#C8C8C8"
    base_edge = "#8A8A8A"
    ink = "#2B2B2B"
    mute = "#8A8A8A"

    panels = [
        ("acc3", r"Acc$_3$ $\uparrow$", "higher is better", False),
        ("acc5", r"Acc$_5$ $\uparrow$", "higher is better", False),
        ("mae", r"MAE $\downarrow$", "lower is better", True),
        ("rho", r"Spearman $\rho$ $\uparrow$", "higher is better", False),
    ]
    letters = ["(a)", "(b)", "(c)", "(d)"]

    n = len(labels)
    # B1 at top … Ours at bottom
    y = np.arange(n - 1, -1, -1)
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.25), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.15, wspace=0.28)

    for ax, (key, title, better, lower_better), letter in zip(axes, panels, letters):
        v = vals[key]
        is_ours = np.array([lab == "Ours" for lab in labels])

        ax.scatter(
            v[~is_ours],
            y[~is_ours],
            s=42,
            c=base_c,
            edgecolors=base_edge,
            linewidths=0.6,
            zorder=2,
        )
        ax.scatter(
            v[is_ours],
            y[is_ours],
            s=64,
            c=ours_c,
            edgecolors=ours_c,
            linewidths=0.8,
            zorder=3,
        )

        ours_i = int(np.where(is_ours)[0][0])
        base_idx = np.where(~is_ours)[0]
        best_b = int(base_idx[np.argmin(v[base_idx])] if lower_better else base_idx[np.argmax(v[base_idx])])

        def _fmt(x: float) -> str:
            return f"{x:.3f}"

        lo, hi = float(v.min()), float(v.max())
        span = max(hi - lo, 1e-3)

        # Ours value clearly to the right of the marker (not on top of it).
        ax.annotate(
            _fmt(v[ours_i]),
            (v[ours_i], y[ours_i]),
            textcoords="offset points",
            xytext=(11, 0),
            va="center",
            ha="left",
            fontsize=7.5,
            color=ours_c,
            fontweight="bold",
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.97),
        )

        ax.set_xlim(lo - 0.10 * span, hi + 0.38 * span)
        ax.set_ylim(-0.65, n - 0.35)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)

        ax.set_title(f"{letter}  {title}", fontsize=9, fontweight="bold", color=ink, pad=12)
        ax.text(
            0.5,
            1.015,
            f"best base: {labels[best_b]} {_fmt(v[best_b])}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=mute,
            clip_on=False,
        )
        ax.set_xlabel(better, fontsize=6.5, color=mute, labelpad=3)

        ax.grid(axis="x", color="#E6E6E6", linewidth=0.7, zorder=0)
        ax.grid(axis="y", color="#F2F2F2", linewidth=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CFCFCF")
        ax.spines["bottom"].set_color("#CFCFCF")
        ax.tick_params(length=2.2, colors=ink, labelsize=7.5)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=base_c, markeredgecolor=base_edge, markersize=7, label="Baselines B1–B6"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ours_c, markeredgecolor=ours_c, markersize=8, label="Ours (PASS-Ord)"),
    ]
    fig.suptitle("Nested interleaved evaluation", fontweight="bold", fontsize=10.5, color=ink, y=0.98)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.915), fontsize=7.5)
    save(fig, "Fig03_main_comparison")


def fig04_ablation():
    """One combined ablation figure: Acc3 / Acc5 / MAE side by side; labels clear of the Full line."""
    df = pd.read_csv(TAB / "T6_ablation.csv").set_index("variant")

    remove_keys = [
        "Ours_v9_full_main",
        "w/o_ResNet_progress",
        "w/o_task_routing",
        "w/o_RF_in_cls",
    ]
    single_keys = [
        "single_expert_LSTM",
        "single_expert_RF",
    ]
    name = {
        "Ours_v9_full_main": "Full",
        "w/o_ResNet_progress": "w/o ResNet prog.",
        "w/o_task_routing": "w/o task routing",
        "w/o_RF_in_cls": "w/o RF in cls",
        "single_expert_LSTM": "LSTM only",
        "single_expert_RF": "RF only (\u2248B5)",
    }

    rows = [("data", "Ours_v9_full_main")]
    rows.append(("header", "Remove component"))
    for k in remove_keys[1:]:
        rows.append(("data", k))
    rows.append(("header", "Single expert"))
    for k in single_keys:
        rows.append(("data", k))

    y_cursor = 0.0
    y_list, kinds, keys_plot, labels = [], [], [], []
    for kind, payload in rows:
        y_list.append(y_cursor)
        kinds.append(kind)
        if kind == "header":
            keys_plot.append(None)
            labels.append(payload)
            y_cursor += 1.05
        else:
            keys_plot.append(payload)
            labels.append(name[payload])
            y_cursor += 1.15
    y = np.asarray(y_list, dtype=float)
    y = y.max() - y  # Full on top
    kinds = np.asarray(kinds)
    is_data = kinds == "data"
    is_header = kinds == "header"
    is_full = np.array([(k == "Ours_v9_full_main") for k in keys_plot])

    vals = {
        "acc3": np.array([float(df.loc[k, "cls3_acc"]) if k else np.nan for k in keys_plot]),
        "acc5": np.array([float(df.loc[k, "cls5_acc"]) if k else np.nan for k in keys_plot]),
        "mae": np.array([float(df.loc[k, "progress_mae"]) if k else np.nan for k in keys_plot]),
    }

    ours_c = "#4A90C8"
    ours_band = "#E8F2FA"
    base_c = "#9A9A9A"
    base_edge = "#6E6E6E"
    ink = "#222222"
    mute = "#6A6A6A"

    panels = [
        ("acc3", r"(a) Acc$_3$ $\uparrow$", "higher is better"),
        ("acc5", r"(b) Acc$_5$ $\uparrow$", "higher is better"),
        ("mae", r"(c) MAE $\downarrow$", "lower is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.985, top=0.86, bottom=0.13, wspace=0.22)

    full_y = float(y[is_full][0])

    for ax, (key, title, better) in zip(axes, panels):
        v = vals[key]
        full_v = float(v[is_full][0])
        data_mask = is_data & ~np.isnan(v)

        ax.axhspan(full_y - 0.42, full_y + 0.42, color=ours_band, zorder=0, lw=0)
        ax.axvline(full_v, color=ours_c, lw=1.1, ls="--", alpha=0.75, zorder=1)

        m_var = data_mask & ~is_full
        ax.scatter(
            v[m_var], y[m_var], s=72, c=base_c,
            edgecolors=base_edge, linewidths=0.8, zorder=3,
        )
        ax.scatter(
            v[is_full], y[is_full], s=105, c=ours_c,
            edgecolors=ours_c, linewidths=1.0, zorder=4,
        )

        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))
        span = max(hi - lo, 1e-3)
        ax.set_xlim(lo - 0.28 * span, hi + 0.28 * span)

        for i in np.where(data_mask)[0]:
            # Beside marker, off the dashed Full line.
            if v[i] >= full_v - 0.08 * span:
                xytext = (-9, 0)
                ha = "right"
            else:
                xytext = (9, 0)
                ha = "left"
            # MAE: four decimals so Full reads 0.0147 (matches nested table), not 0.015.
            val_txt = f"{v[i]:.4f}" if key == "mae" else f"{v[i]:.3f}"
            ax.annotate(
                val_txt,
                (v[i], y[i]),
                textcoords="offset points",
                xytext=xytext,
                va="center",
                ha=ha,
                fontsize=7.8 if is_full[i] else 7.4,
                color=ours_c if is_full[i] else mute,
                fontweight="bold" if is_full[i] else "normal",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95),
                zorder=5,
            )

        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=9.0)
            for tick, hdr, full in zip(ax.get_yticklabels(), is_header, is_full):
                if hdr:
                    tick.set_fontsize(8.0)
                    tick.set_color(mute)
                    tick.set_fontstyle("italic")
                    tick.set_fontweight("bold")
                elif full:
                    tick.set_color(ours_c)
                    tick.set_fontweight("bold")
                    tick.set_fontsize(9.5)
                else:
                    tick.set_color(ink)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=ink, pad=8)
        ax.set_xlabel(better, fontsize=8.0, color=mute, labelpad=4)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.7, zorder=0)
        ax.grid(axis="y", color="#F3F3F3", linewidth=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CFCFCF")
        ax.spines["bottom"].set_color("#CFCFCF")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2.5, colors=ink, labelsize=8.5)
        ax.set_ylim(y.min() - 0.75, y.max() + 0.75)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=ours_c, markeredgecolor=ours_c, markersize=8, label="Full",
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=base_c, markeredgecolor=base_edge, markersize=7, label="Ablation variants",
        ),
        plt.Line2D([0], [0], color=ours_c, lw=1.1, ls="--", label="Full reference"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.58, 0.98), fontsize=8.5,
    )
    save(fig, "Fig04_ablation")


def fig05_confusion():
    """Side-by-side confusion matrices: row-normalized color, count + % in cells."""
    cm3 = pd.read_csv(TAB / "T10_cm_cls3.csv", index_col=0).to_numpy().astype(float)
    cm5 = pd.read_csv(TAB / "T10_cm_cls5.csv", index_col=0).to_numpy().astype(float)

    panels = [
        (cm3, ["Early", "Mid", "Late"], "(a) Three-stage classification"),
        (cm5, [f"Bin {i}" for i in range(1, 6)], "(b) Five-bin classification"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55))
    fig.subplots_adjust(left=0.08, right=0.88, top=0.82, bottom=0.14, wspace=0.28)
    ink = "#222222"
    mute = "#666666"
    err_ink = "#1A1A1A"
    im = None

    for ax, (cm, ticklabels, title) in zip(axes, panels):
        row = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        acc = float(cm.diagonal().sum() / cm.sum())
        n = int(cm.sum())
        n_cls = len(ticklabels)

        im = ax.imshow(row, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xlim(-0.5, n_cls - 0.5)
        ax.set_ylim(n_cls - 0.5, -0.5)
        ax.set_box_aspect(1)

        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(ticklabels, fontsize=7.2)
        ax.set_yticklabels(ticklabels, fontsize=7.2)
        ax.set_xlabel("Predicted", fontsize=8.0, color=ink, labelpad=3)
        ax.set_ylabel("True", fontsize=8.0, color=ink, labelpad=3)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=ink, pad=18)
        ax.text(
            0.5, 1.02,
            f"Acc = {acc:.3f}  ·  n = {n} passes",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.0, color=mute, clip_on=False,
        )

        fs_count = 8.2 if n_cls <= 3 else 6.6
        fs_pct = 6.6 if n_cls <= 3 else 5.4

        for i in range(n_cls):
            for j in range(n_cls):
                v = int(cm[i, j])
                if v == 0:
                    continue
                r = float(row[i, j])
                pct = 100.0 * r
                on_diag = i == j
                if on_diag and r >= 0.45:
                    color = "white"
                    weight = "bold"
                else:
                    color = err_ink
                    weight = "bold" if not on_diag else "normal"
                ax.text(
                    j, i - 0.14, str(v),
                    ha="center", va="center", color=color,
                    fontsize=fs_count, fontweight=weight, clip_on=True,
                )
                ax.text(
                    j, i + 0.20, f"{pct:.1f}%",
                    ha="center", va="center", color=color,
                    fontsize=fs_pct, fontweight="normal", clip_on=True,
                )

        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
            sp.set_color("#B0B0B0")
        ax.tick_params(length=0, colors=ink)
        ax.set_xticks(np.arange(-0.5, n_cls, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_cls, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)

    cax = fig.add_axes([0.90, 0.18, 0.018, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized rate", fontsize=7.5, color=ink)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=6.5, length=2.0)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#B0B0B0")
    save(fig, "Fig05_confusion")


def fig06_progress(pred_path: Path | None = None):
    """Left: per-grind metal-exposure cue; right: frame-level progress prediction."""
    from matplotlib.ticker import FuncFormatter
    from scipy.stats import spearmanr

    cache = OUT / "_progress_points.npz"
    seg = pd.read_csv(SEG_CSV)

    # Matched panel heights; right panel stays square via equal aspect.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.45))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.15, wspace=0.32)

    light_blue = "#A8C8E8"
    deep_blue = "#2F6FAE"
    orange = "#C47A00"
    ink = "#222222"
    mute = "#555555"
    fs_ax = 8.5
    fs_tick = 7.5
    fs_title = 10.0
    lw_trend = 1.7
    lw_ref = 1.15

    # ----- (a) Metal exposure trend (per grind) -----
    ax = axes[0]
    g = seg["grind"].to_numpy(dtype=float)
    y = seg["metal_ratio"].to_numpy(dtype=float) * 100.0  # percent
    order = np.argsort(g)
    g_s, y_s = g[order], y[order]

    ax.scatter(
        g_s,
        y_s,
        s=14,
        c=light_blue,
        alpha=0.75,
        edgecolors="none",
        zorder=2,
        label="Observed (per grind)",
    )
    # Edge-safe centered rolling mean (avoids zero-pad drop of np.convolve mode='same')
    win = 7
    y_sm = (
        pd.Series(y_s)
        .rolling(window=win, center=True, min_periods=max(1, win // 2))
        .mean()
        .to_numpy()
    )
    ax.plot(g_s, y_sm, color=deep_blue, lw=lw_trend, zorder=3, label="Smoothed trend")

    g_star = 94
    ymax = float(np.nanmax(y_s))
    y_hi = max(25.0, np.ceil(ymax / 5.0) * 5.0)
    ax.set_xlim(0, 175)
    ax.set_ylim(0, y_hi)
    ax.axvline(g_star, color=orange, ls="--", lw=lw_ref, zorder=4)
    # Keep label inside axes (left of vline, mid-height) so it never collides with the title
    ax.text(
        g_star - 3.5,
        12.5,
        r"Reference" + "\n" + r"$g\approx 94$",
        ha="right",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=ink,
        linespacing=1.25,
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=orange, lw=0.9, alpha=0.96),
    )

    ax.set_xlabel("Grind index $g$", fontsize=fs_ax, color=ink)
    ax.set_ylabel("Metal exposure ratio (%)", fontsize=fs_ax, color=ink)
    ax.set_title("(a) Metal exposure trend", fontsize=fs_title, fontweight="bold", color=ink, pad=8)
    ax.set_yticks(np.arange(0, y_hi + 1e-9, 5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)}%"))
    ax.tick_params(labelsize=fs_tick, colors=ink, length=2.5)
    ax.grid(axis="y", color="#E8E8E8", lw=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.legend(frameon=False, loc="lower right", fontsize=7.4, handlelength=1.8)
    # ----- (b) Progress prediction (frame level) -----
    ax = axes[1]
    if cache.exists():
        d = np.load(cache)
        yt, yp = np.asarray(d["yt"], dtype=float), np.asarray(d["yp"], dtype=float)
        ax.scatter(
            yt,
            yp,
            s=4.5,
            c=deep_blue,
            alpha=0.12,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        lim = [0.0, 1.0]
        ax.plot(lim, lim, color=orange, lw=lw_ref, ls="--", zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        mae = float(np.mean(np.abs(yt - yp)))
        sp = float(spearmanr(yt, yp).correlation)
        ax.text(
            0.04,
            0.96,
            f"MAE = {mae:.3f}\n"
            + r"Spearman $\rho$ = "
            + f"{sp:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            color=ink,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#D0D0D0", lw=0.6, alpha=0.95),
        )
        n_fr = len(yt)
        ax.text(
            0.98,
            0.03,
            f"frame-level  ·  n = {n_fr}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            color=mute,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
        )
    else:
        ax.text(0.5, 0.5, "Progress points unavailable", ha="center", va="center", color=mute)

    ax.set_xlabel(r"Reference progress ($g/169$)", fontsize=fs_ax, color=ink)
    ax.set_ylabel("Predicted progress", fontsize=fs_ax, color=ink)
    ax.set_title("(b) Progress prediction", fontsize=fs_title, fontweight="bold", color=ink, pad=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.tick_params(labelsize=fs_tick, colors=ink, length=2.5)
    ax.grid(color="#E8E8E8", lw=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")

    save(fig, "Fig06_progress_and_metal")


def fig07_qualitative():
    """Classic 3×4 panorama grid (matched pan/overlay/mask per column)."""
    pan_seg = ROOT / "outputs/processed/segmentation/panorama_seg"
    stages = [
        ("grind_001_pass0.png", "grind_001_pass0_overlay.png", "grind_001_pass0_mask.png", "g=1"),
        ("grind_050_pass2.png", "grind_050_pass2_overlay.png", "grind_050_pass2_mask.png", "g=50"),
        ("grind_100_pass0.png", "grind_100_pass0_overlay.png", "grind_100_pass0_mask.png", "g=100"),
        ("grind_169_pass0.png", "grind_169_pass0_overlay.png", "grind_169_pass0_mask.png", "g=169"),
    ]

    # Unify to a shared display size while keeping pan/overlay/mask aligned within each column.
    out_w, out_h = 280, 320

    def _load_triplet(pan_fn, ov_fn, mk_fn):
        pan = np.asarray(Image.open(PAN / pan_fn).convert("RGB"))
        ov = np.asarray(Image.open(pan_seg / ov_fn).convert("RGB"))
        mk = np.asarray(Image.open(pan_seg / mk_fn).convert("L"))
        assert pan.shape[:2] == ov.shape[:2] == mk.shape, (pan_fn, pan.shape, ov.shape, mk.shape)
        pan = np.asarray(Image.fromarray(pan).resize((out_w, out_h), Image.Resampling.BICUBIC))
        ov = np.asarray(Image.fromarray(ov).resize((out_w, out_h), Image.Resampling.BICUBIC))
        mk = np.asarray(Image.fromarray(mk).resize((out_w, out_h), Image.Resampling.NEAREST))
        return pan, ov, mk

    fig, axes = plt.subplots(3, 4, figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.04, wspace=0.06, hspace=0.08)

    row_names = ["Panorama", "Seg. overlay", "Metal mask"]
    for c, (pan_fn, ov_fn, mk_fn, lab) in enumerate(stages):
        pan, ov, mk = _load_triplet(pan_fn, ov_fn, mk_fn)
        panels = [pan, ov, mk]
        for r in range(3):
            ax = axes[r, c]
            if r == 2:
                ax.imshow(panels[r], cmap="gray", vmin=0, vmax=255, aspect="auto")
            else:
                ax.imshow(panels[r], aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(lab, fontsize=9, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(row_names[r], fontsize=8.5, fontweight="bold", labelpad=6)

    fig.suptitle(
        "Qualitative grinding progression & weak-supervision masks",
        fontweight="bold",
        fontsize=10.5,
        y=0.97,
    )
    save(fig, "Fig07_qualitative")


def fig08_extras():
    seed = pd.read_csv(TAB / "seed_mean_std.csv").iloc[0]
    per = pd.read_csv(TAB / "seed_v9_per_seed.csv")
    wdf = pd.read_csv(TAB / "A_weight_sensitivity.csv")
    e2e = pd.read_csv(TAB / "T8_e2e_latency.csv")
    sig = pd.read_csv(TAB / "T_significance.csv")

    ink = "#222222"
    mute = "#777777"
    ours_c = "#4A90C8"
    orange = "#E69F00"
    seed_c = "#B0B0B0"

    # Compatible with nested 4-seed ResNet schema or legacy 8-seed columns
    def _seed_get(row, *names, default=None):
        for n in names:
            if n in row.index:
                return float(row[n])
        return default

    n_seeds = len(per)
    cls3_mu = _seed_get(seed, "cls3_acc_mean", "cls3_mean")
    cls3_sd = _seed_get(seed, "cls3_acc_std", "cls3_std")
    cls5_mu = _seed_get(seed, "cls5_acc_mean", "cls5_mean")
    cls5_sd = _seed_get(seed, "cls5_acc_std", "cls5_std")
    sp_mu = _seed_get(seed, "progress_spearman_mean", "spearman_mean")
    sp_sd = _seed_get(seed, "progress_spearman_std", "spearman_std")
    ens_c3 = _seed_get(seed, "ensemble_cls3", "primary_ens_cls3")
    ens_c5 = _seed_get(seed, "ensemble_cls5", "primary_ens_cls5")
    ens_sp = _seed_get(seed, "ensemble_spearman_frame", "primary_ens_spearman", "progress_spearman_mean")

    # Compact 3-panel row: avoid large top/bottom margins from outside legends.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.8, 2.45),
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.95]},
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.22, wspace=0.42)

    # ----- (a) Multi-seed -----
    ax = axes[0]
    ens_dx = 0.26  # identical Ensemble offset for all three metrics
    seed_half = 0.15
    metric_cols = [
        ("cls3_acc", "Acc$_3$", cls3_mu, cls3_sd, ens_c3),
        ("cls5_acc", "Acc$_5$", cls5_mu, cls5_sd, ens_c5),
        ("progress_spearman", r"Spearman $\rho$", sp_mu, sp_sd, ens_sp),
    ]
    for i, (col, _lab, mu, sd, ens_v) in enumerate(metric_cols):
        vals = per[col].to_numpy(float)
        xs = np.full(len(vals), float(i), dtype=float)
        xs += np.linspace(-seed_half, seed_half, len(vals))
        rounded = np.round(vals, 6)
        for uv in np.unique(rounded):
            idx = np.where(rounded == uv)[0]
            if len(idx) > 1:
                xs[idx] += np.linspace(-0.035, 0.035, len(idx))
        ax.scatter(xs, vals, s=20, c=seed_c, edgecolors="#7A7A7A", linewidths=0.35, zorder=2, alpha=0.95)
        ax.errorbar(
            i,
            float(mu),
            yerr=float(sd),
            fmt="none",
            ecolor="#444444",
            elinewidth=1.2,
            capsize=3.2,
            zorder=3,
        )
        ax.scatter([i], [float(mu)], s=30, c="#333333", zorder=4, edgecolors="none")
        ax.scatter(
            [i + ens_dx],
            [float(ens_v)],
            s=52,
            marker="D",
            c=ours_c,
            edgecolors="none",
            zorder=5,
        )

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([m[1] for m in metric_cols], fontsize=7.8)
    ax.set_xlim(-0.55, 2.65)
    y0 = float(min(per["cls3_acc"].min(), per["cls5_acc"].min(), per["progress_spearman"].min()))
    ax.set_ylim(max(0.93, y0 - 0.01), 1.002)
    ax.set_ylabel("Score", fontsize=8, color=ink, labelpad=4)
    ax.set_title(f"(a) Multi-seed ($n{{=}}{n_seeds}$)", fontsize=9.2, fontweight="bold", color=ink, pad=4)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=seed_c, markersize=5.5, label="Seeds"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#333333",
            markersize=5.5,
            label=r"Mean$\pm$SD",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=ours_c,
            markersize=6.0,
            label="Ensemble",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=6.2,
        columnspacing=0.8,
        handletextpad=0.25,
        borderaxespad=0.2,
    )
    ax.tick_params(labelsize=7.2, colors=ink, length=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8C8C8")
    ax.spines["bottom"].set_color("#C8C8C8")

    # ----- (b) Acc5 soft-blend diagnostic vs maxconf -----
    ax = axes[1]
    if "a5_rf" in wdf.columns:
        soft = wdf[wdf["a5_rf"] >= 0].sort_values("a5_rf")
        ax.plot(soft["a5_rf"], soft["cls5_acc"], "-", color=ours_c, lw=1.35, zorder=2)
        ax.scatter(
            soft["a5_rf"],
            soft["cls5_acc"],
            s=28,
            facecolors="white",
            edgecolors=ours_c,
            linewidths=1.15,
            zorder=3,
        )
        main = wdf[wdf["note"].astype(str).str.contains("maxconf", case=False, na=False)]
        wy = float(main["cls5_acc"].iloc[0]) if len(main) else float(soft["cls5_acc"].max())
        if len(main):
            ax.axhline(wy, color=orange, ls="--", lw=0.9, alpha=0.9, zorder=1)
            ax.text(
                0.02,
                0.98,
                f"maxconf Acc$_5$={wy:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                color=orange,
                fontweight="bold",
                clip_on=False,
            )
        ax.set_xlabel(r"soft Acc$_5$ weight $w_{\mathrm{RF}}$", fontsize=8, color=ink)
        ax.set_ylabel(r"Acc$_5$ $\uparrow$", fontsize=8, color=ink)
        ax.set_title("(b) Acc$_5$ fusion", fontsize=9.2, fontweight="bold", color=ink, pad=4)
        ymin = float(soft["cls5_acc"].min()) - 0.004
        ymax = float(max(soft["cls5_acc"].max(), wy)) + 0.006
        ax.set_ylim(ymin, ymax)
    else:
        sub = wdf[wdf.perturb.str.startswith("resnet")].drop_duplicates(subset=["w_resnet"]).copy()
        sub = sub.sort_values("w_resnet")
        ax.plot(sub["w_resnet"], sub["progress_mae"], "-", color=ours_c, lw=1.35, zorder=2)
        ax.scatter(
            sub["w_resnet"],
            sub["progress_mae"],
            s=28,
            facecolors="white",
            edgecolors=ours_c,
            linewidths=1.15,
            zorder=3,
        )
        ax.set_xlabel(r"$w_{\mathrm{ResNet}}$", fontsize=8, color=ink)
        ax.set_ylabel("MAE $\\downarrow$", fontsize=8, color=ink)
        ax.set_title("(b) Weight sensitivity", fontsize=9.2, fontweight="bold", color=ink, pad=4)
    ax.tick_params(labelsize=7.2, colors=ink, length=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8C8C8")
    ax.spines["bottom"].set_color("#C8C8C8")

    # ----- (c) Latency (short y-labels; full names in caption) -----
    ax = axes[2]
    e2e_i = e2e.set_index("stage")
    labs = ["Cached", "Full path"]
    vals = np.array(
        [
            float(e2e_i.loc["E2E features-ready (RF+LSTM+fuse)", "ms_per_pass"]),
            float(e2e_i.loc["E2E image→result (4帧 ResNet + LSTM + RF + fuse)", "ms_per_pass"]),
        ]
    )
    y = np.arange(len(labs), dtype=float)
    ax.barh(y, vals, color=[ours_c, "#9BBFDC"], height=0.55, left=0.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7.8)
    ax.set_ylim(1.15, -0.15)  # tight vertical span (inverted)
    ax.set_xlim(0, max(88.0, float(vals.max()) + 14.0))
    for yi, v in zip(y, vals):
        ax.text(v + 1.8, yi, f"{v:.2f}", va="center", ha="left", fontsize=7.5, color=ink, fontweight="bold")
    ax.set_xlabel("ms / pass", fontsize=8, color=ink)
    ax.set_title("(c) Latency", fontsize=9.2, fontweight="bold", color=ink, pad=4)
    ax.tick_params(labelsize=7.2, colors=ink, length=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8C8C8")
    ax.spines["bottom"].set_color("#C8C8C8")

    save(fig, "Fig08_seed_weight_latency", pad_inches=0.02)

    # significance: nested Acc3 vs B3 / Acc5 vs B5
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 2.85),
        gridspec_kw={"width_ratios": [1.05, 1.15]},
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.20, wspace=0.55)

    if "contrast" in sig.columns:
        acc = sig[sig.contrast.astype(str).str.startswith("Acc")].copy()
        mae = sig[sig.contrast.astype(str).str.contains("MAE", case=False)].copy()

        # (a) Acc point estimates Ours vs baseline
        ax = axes[0]
        rows_plot = []
        for _, r in acc.iterrows():
            tag = str(r.contrast).replace("Acc3_vs_B3", "Acc$_3$").replace("Acc5_vs_B5", "Acc$_5$")
            rows_plot.append((f"Ours {tag}", float(r.ours_acc), True))
            base = "B3" if "Acc3" in str(r.contrast) else "B5"
            rows_plot.append((f"{base} {tag}", float(r.base_acc), False))
        names = [t[0] for t in rows_plot]
        y = np.arange(len(names), dtype=float)[::-1]
        vals = np.array([t[1] for t in rows_plot])
        colors = [C["ours"] if t[2] else C["base"] for t in rows_plot]
        ax.scatter(vals, y, c=colors, s=44, zorder=3, edgecolors="none")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.6)
        ax.set_ylim(y.min() - 0.55, y.max() + 0.55)
        ax.set_xlabel("Accuracy", fontsize=8)
        ax.set_xlim(0.94, 1.005)
        ax.set_title("(a) Acc point estimates", fontweight="bold", fontsize=9.2, pad=6)
        ax.tick_params(labelsize=7.0, length=2.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C["light"])
        ax.spines["bottom"].set_color(C["light"])

        # (b) Bootstrap deltas (Acc and frame MAE)
        ax = axes[1]
        names, vals, lo, hi, ptxt = [], [], [], [], []
        label_map = {
            "Acc3_vs_B3": r"Acc$_3$ vs B3",
            "Acc5_vs_B5": r"Acc$_5$ vs B5",
        }
        for _, r in acc.iterrows():
            key = str(r.contrast)
            names.append(label_map.get(key, key.replace("_", " ")))
            vals.append(float(r.delta))
            lo.append(float(r.delta_ci_lo))
            hi.append(float(r.delta_ci_hi))
            p = float(r.bootstrap_p_one_sided)
            ptxt.append(f"$p$={p:.3f}")
        if len(mae):
            r = mae.iloc[0]
            names.append(r"MAE (B3$-$Ours)")
            vals.append(float(r.delta_mae_base_minus_ours))
            lo.append(float(r.delta_ci_lo))
            hi.append(float(r.delta_ci_hi))
            ptxt.append(f"$p$={float(r.bootstrap_p_one_sided):.2g}")
        y = np.arange(len(names), dtype=float)[::-1]
        vals, lo, hi = map(np.asarray, (vals, lo, hi))
        ax.axvline(0, color="#B0B0B0", lw=0.9, zorder=1)
        ax.hlines(y, lo, hi, color=C["blue"], lw=1.7, zorder=2)
        ax.scatter(vals, y, c=C["ours"], s=44, zorder=3, edgecolors="none")
        # Place p to the right of CI with enough gap; keep inside axes
        x_max = float(max(hi.max() + 0.006, 0.028))
        for i, yi in enumerate(y):
            ax.text(
                float(hi[i]) + 0.0012,
                yi + 0.18,
                ptxt[i],
                va="bottom",
                ha="left",
                fontsize=6.6,
                color="#555555",
                clip_on=False,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.4)
        ax.set_ylim(y.min() - 0.65, y.max() + 0.65)
        ax.set_xlim(min(-0.004, float(lo.min()) - 0.002), x_max)
        ax.set_xlabel(r"Bootstrap $\Delta$ (95% CI)", fontsize=8)
        ax.set_title("(b) Paired bootstrap tests", fontweight="bold", fontsize=9.2, pad=6)
        ax.tick_params(labelsize=7.0, length=2.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C["light"])
        ax.spines["bottom"].set_color(C["light"])
        fig.suptitle("Nested even/odd hold-out: Ours vs B3/B5", fontweight="bold", fontsize=10, y=0.98)
    else:
        # legacy schema
        ax = axes[0]
        acc_rows = sig[sig.metric == "cls3_acc"].copy()
        names = [
            str(r.method).replace("B5_MeanPool_RF", "B5").replace("B6_BiLSTM", "B6").replace("Ours_v9", "Ours")
            for _, r in acc_rows.iterrows()
        ]
        y = np.arange(len(names))[::-1]
        vals = acc_rows["value"].to_numpy(float)
        lo = acc_rows["ci95_lo"].to_numpy(float)
        hi = acc_rows["ci95_hi"].to_numpy(float)
        colors = [C["ours"] if "Ours" in n else C["base"] for n in names]
        ax.hlines(y, lo, hi, color=C["light"], lw=1.5)
        ax.scatter(vals, y, c=colors, s=36, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("cls3 Acc (95% CI)")
        ax.set_title("(a) Accuracy with bootstrap CI", fontweight="bold")
        ax = axes[1]
        mc = sig[sig.metric == "cls3_McNemar"].copy()
        names = [str(r.method).replace("Ours_vs_", "vs ") for _, r in mc.iterrows()]
        y = np.arange(len(names))[::-1]
        vals = mc["value"].to_numpy(float)
        lo = mc["ci95_lo"].to_numpy(float)
        hi = mc["ci95_hi"].to_numpy(float)
        ax.axvline(0, color=C["light"], lw=0.8)
        ax.hlines(y, lo, hi, color=C["blue"], lw=1.6)
        ax.scatter(vals, y, c=C["ours"], s=40, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel(r"$\Delta$Acc$_3$ (Ours $-$ baseline)")
        ax.set_title("(b) McNemar paired tests", fontweight="bold")
        fig.suptitle("Statistical tests vs B5/B6 (interleaved)", fontweight="bold", y=0.98)

    save(fig, "Fig09_significance", pad_inches=0.03)


def dump_progress_points(device="cuda:0"):
    """Cache frame-level true/pred progress for Fig06."""
    import torch
    from training.features import load_feature_cache
    from training.fill_v9_gaps import eval_specialists, MAIN_PROG_W
    from training.fill_v9_extras import ens_specialists, load_frame_pp
    from training.polish_full_sota import load_seg, train_rf, ens, ROOT as R
    from training.labels import progress

    print("Dumping progress points...", flush=True)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    X, meta, names = load_feature_cache(R / "outputs/baselines/features")
    seg = load_seg(R / "outputs/processed/segmentation/grind_seg_metrics.csv")
    rf = train_rf(X, meta, "interleaved")
    frame_pp = load_frame_pp("interleaved")
    ckpts = sorted((R / "outputs/baselines/ours_v8").glob("ours_v8_interleaved_s*.pt"))
    e_c3, e_c5, e_mae = ens_specialists(X, meta, names, seg, "interleaved", device_t, ckpts)
    a, b, c = MAIN_PROG_W["resnet"], MAIN_PROG_W["rf"], MAIN_PROG_W["lstm"]
    yt, yp = [], []
    for i, r in enumerate(e_c3):
        key = r["key"]
        for rv in frame_pp[key]:
            yt.append(progress(key[0]))
            yp.append(a * rv + b * rf[key]["pp"] + c * e_mae[i]["pp"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "_progress_points.npz", yt=np.asarray(yt), yp=np.asarray(yp))
    print(f"  saved {len(yt)} frame points", flush=True)


def write_index():
    text = """# Paper figures (Sensors / top-venue style)

All figures: PDF (vector) + PNG @ 300 dpi in this folder.

| File | Suggested placement | Content |
|------|---------------------|---------|
| `Fig01_overview` | Fig. 1 | Problem, labels, interleaved protocol + grinding appearance strip |
| `Fig02_method_PASSOrd` | Fig. 2 | PASS-Ord task-routed multi-expert schematic |
| `Fig03_main_comparison` | Fig. 3 | Main table metrics vs B1–B6 (interleaved) |
| `Fig04_ablation` | Fig. 4 | Acc3 / Acc5 / MAE ablation (combined) |
| `Fig05_confusion` | Fig. 5 | cls3/cls5 confusion matrices (combined) |
| `Fig06_progress_and_metal` | Fig. 6 | Metal-ratio cue + progress scatter |
| `Fig07_qualitative` | Fig. 7 | Panorama / overlay / mask progression |
| `Fig08_seed_weight_latency` | Fig. 8 or Appendix | Multi-seed, weight sensitivity, E2E latency |
| `Fig09_significance` | Appendix table-fig | McNemar / bootstrap CI summary |

Style: Okabe–Ito palette, DejaVu Sans, no chartjunk, spines reduced, PDF fonttype 42.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print("  wrote README.md", flush=True)


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Paper figures ===", flush=True)
    fig01_overview()
    fig02_method()
    fig03_main()
    fig04_ablation()
    fig05_confusion()
    # progress points
    if not (OUT / "_progress_points.npz").exists():
        try:
            dump_progress_points()
        except Exception as e:
            print("progress dump failed:", e, flush=True)
    fig06_progress()
    fig07_qualitative()
    fig08_extras()
    write_index()
    # regenerate fig06 if points now exist
    if (OUT / "_progress_points.npz").exists():
        fig06_progress()
    print("DONE →", OUT, flush=True)


if __name__ == "__main__":
    main()
