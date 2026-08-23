"""Generates the thesis figures for docs/PROJECT_REPORT.md as print-ready PNGs.

Every number plotted here is read from a measured artefact -- metrics_*.json
(eval_full_metrics.py), face_hypothesis.json (analyze_face_hypothesis.py), or
a constant block below whose values are quoted directly from the report and
traceable to a training run. Nothing is illustrative or approximated: if a
figure can't be built from a real measurement it isn't built.

Output: docs/figures/*.png at 300 dpi.

Run:  python make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# Validated categorical palette (scripts/validate_palette.js, light mode,
# --pairs all: all checks PASS). Slot 3 sits below 3:1 contrast on this
# surface, so the "relief rule" applies -- every chart using it carries
# visible direct labels, which is why value labels below are mandatory
# rather than decorative.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def style(ax, ylabel=None, pct=True):
    """Recessive axes: no top/right spines, horizontal grid only."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.xaxis.grid(False)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


def label_bars(ax, bars, values, fmt="{:.1%}", dy=0.012):
    """Direct value labels -- required by the palette's relief rule, and
    the reason no reader has to measure a bar against the axis."""
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="medium")


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 5.1 / 5.2
def category_figure(metrics, title, fname, note):
    cats = [("word", "Words"), ("digit", "Digits"), ("letter", "Letters")]
    cats = [(k, lab) for k, lab in cats if k in metrics["by_category"]]
    # Two bars per category, because one would misrepresent the result.
    # The unmasked bar is the model metric: argmax over every class at
    # once. The in-app bar is what the product actually does, restricting
    # the argmax to the mode the user selected. For BdSL digits the two
    # differ by nearly 29 points -- the digits were losing to *word*
    # classes they can never meet in Digits mode -- so plotting only the
    # unmasked number would describe a failure the user never experiences,
    # and plotting only the in-app number would flatter the model.
    vals = [metrics["by_category"][k]["accuracy"] for k, _ in cats]
    in_app = [metrics["by_category"][k].get("accuracy_in_app_masked") for k, _ in cats]
    has_in_app = all(v is not None for v in in_app)
    labels = [f"{lab}\n{metrics['by_category'][k]['n_classes']} classes  ·  n={metrics['by_category'][k]['n']:,}"
              for k, lab in cats]
    agg = metrics["overall"]["accuracy"]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = range(len(vals))
    if has_in_app:
        w = 0.34
        b1 = ax.bar([i - w / 2 for i in x], vals, w, color=BLUE, zorder=3,
                    label="All classes competing (model metric)")
        b2 = ax.bar([i + w / 2 for i in x], in_app, w, color=ORANGE, zorder=3,
                    label="Restricted to the active mode (as shipped)")
        label_bars(ax, b1, vals)
        label_bars(ax, b2, in_app)
        # Legend directly above the plot, subtitle above the legend. The
        # two-row legend is ~0.13 axes units tall, so the subtitle has to
        # clear 1.14 -- at the default it landed on top of the legend text.
        ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK_2,
                  bbox_to_anchor=(0, 1.005), ncol=1)
    else:
        b1 = ax.bar(list(x), vals, 0.58, color=BLUE, zorder=3)
        label_bars(ax, b1, vals)

    ax.axhline(agg, color=INK_3, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(len(vals) - 0.55, agg + 0.015, f"combined model: {agg:.1%}",
            ha="right", va="bottom", fontsize=9, color=INK_3, fontweight="medium")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    style(ax, "Top-1 accuracy (held-out)")
    ax.set_title(title, fontsize=12.5, color=INK, fontweight="semibold", loc="left",
                 pad=82 if has_in_app else 30)
    ax.text(0, 1.175 if has_in_app else 1.02, note, transform=ax.transAxes,
            fontsize=9, color=INK_3, va="bottom")
    save(fig, fname)


# ------------------------------------------------------------------- 5.3
def ablation_figure():
    groups = ["BdSL\n(107 classes)", "ASL words\n(250 classes)"]
    single = [0.776, 0.520]
    two = [0.834, 0.529]
    second_hand = ["2nd hand in ~43% of frames", "2nd hand in ~0.1% of frames"]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = range(len(groups))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], single, w, label="One hand", color=BLUE, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], two, w, label="Two hands", color=ORANGE, zorder=3)
    label_bars(ax, b1, single)
    label_bars(ax, b2, two)

    for i, (s, t) in enumerate(zip(single, two)):
        ax.annotate(f"{(t - s) * 100:+.1f} pts", xy=(i, max(s, t) + 0.075), ha="center",
                    fontsize=9.5, color=INK, fontweight="semibold")

    # The second-hand coverage belongs with the group label, not painted
    # inside the bars: grey caption text over a saturated fill failed the
    # contrast check, and it is metadata about the group anyway.
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{g}\n{s}" for g, s in zip(groups, second_hand)])
    ax.set_ylim(0, 1.0)
    style(ax, "Top-1 accuracy (held-out)")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK_2)
    ax.set_title("Two-hand tracking helps only where the corpus recorded a second hand",
                 fontsize=12, color=INK, fontweight="semibold", loc="left", pad=30)
    # Both bars predate the position anchor, deliberately: swapping the
    # later, better number into one side would turn a controlled
    # one-variable ablation into a comparison of two things at once.
    ax.text(0, 1.02, "Measured before the position anchor was added, so the two bars differ by one variable only",
            transform=ax.transAxes, fontsize=9, color=INK_3, va="bottom")
    save(fig, "fig_5_3_twohand_ablation.png")


# ------------------------------------------------------------------- 5.4
def defect_figure():
    stages = ["Single-hand\nbaseline", "Two-hand\n+ Defect 2", "After fixing\nDefect 2",
              "After fixing\nDefect 3"]
    vals = [0.520, 0.465, 0.472, 0.529]
    colors = [INK_3, "#e34948", "#e34948", AQUA]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(range(len(vals)), vals, color=INK_3, linewidth=2, zorder=3, marker="o",
            markersize=9, markerfacecolor=SURFACE, markeredgewidth=2)
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.plot(i, v, marker="o", markersize=9, color=c, zorder=4)
        ax.annotate(f"{v:.1%}", xy=(i, v), xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=10, color=INK, fontweight="semibold")

    ax.axhline(0.520, color=INK_3, linestyle=":", linewidth=1.4, zorder=2)
    ax.text(3.35, 0.521, "baseline", ha="right", va="bottom", fontsize=9, color=INK_3)
    ax.axhspan(0.44, 0.520, color="#e34948", alpha=0.055, zorder=1)
    ax.text(0.5, 0.4485, "below baseline — shipped as an 'improvement' if unverified",
            fontsize=8.8, color="#b3383a", va="bottom")

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0.44, 0.56)
    style(ax, "ASL words top-1 accuracy")
    ax.set_title("Two defects drove accuracy below baseline before being caught",
                 fontsize=12, color=INK, fontweight="semibold", loc="left", pad=14)
    save(fig, "fig_5_4_defect_trajectory.png")


# ------------------------------------------------------------------- 5.5
def confidence_figure():
    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]
    bdsl = {"Words": [1.00, 1.00, 0.99, 0.98, 0.97, 0.95, 0.92],
            "Digits": [0.99, 0.99, 0.90, 0.78, 0.68, 0.55, 0.34],
            "Letters": [1.00, 0.98, 0.93, 0.85, 0.79, 0.71, 0.62]}
    asl = {"Words": [0.94, 0.87, 0.78, 0.73, 0.68, 0.58, 0.47],
           "Digits": [1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 0.99],
           "Letters": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.99]}
    colors = {"Words": BLUE, "Digits": ORANGE, "Letters": AQUA}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True)
    for ax, (name, data) in zip(axes, [("BdSL", bdsl), ("ASL", asl)]):
        for series, vals in data.items():
            ax.plot(thresholds, vals, linewidth=2, color=colors[series], label=series,
                    marker="o", markersize=5, zorder=3)
        ax.axvline(0.55, color=INK_3, linestyle="--", linewidth=1.2, zorder=2)
        ax.axvline(0.40, color=INK_3, linestyle=":", linewidth=1.2, zorder=2)
        ax.set_xlabel("Confidence threshold", color=INK_2)
        ax.set_title(name, fontsize=11.5, color=INK, fontweight="semibold", loc="left")
        style(ax, None)
        ax.set_ylim(0.25, 1.04)
    axes[0].set_ylabel("Fraction of CORRECT predictions kept", color=INK_2)
    axes[0].text(0.552, 0.285, "0.55", fontsize=8.5, color=INK_3)
    axes[0].text(0.402, 0.285, "0.40", fontsize=8.5, color=INK_3)
    axes[1].legend(frameon=False, loc="lower left", fontsize=9.5, labelcolor=INK_2)
    fig.suptitle("A single flat threshold discards correct digit and letter recognitions",
                 fontsize=12, color=INK, fontweight="semibold", x=0.078, ha="left", y=1.02)
    save(fig, "fig_5_5_confidence_tradeoff.png")


# ------------------------------------------------------------------- 5.6
def face_figure(fh):
    groups = [f"Face-located words\nn = {fh['face_located']['n']}",
              f"All other words\nn = {fh['other_words']['n']}"]
    means = [fh["face_located"]["mean_recall"], fh["other_words"]["mean_recall"]]
    medians = [fh["face_located"]["median_recall"], fh["other_words"]["median_recall"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(groups, means, color=[ORANGE, BLUE], width=0.5, zorder=3)
    for i, m in enumerate(medians):
        ax.plot([i - 0.25, i + 0.25], [m, m], color=INK, linewidth=2, zorder=5)
        ax.text(i + 0.27, m, f"median {m:.1%}", va="center", fontsize=8.8, color=INK_2)
    # Clear the median tick as well as the bar top, so the mean label never
    # lands on the line (the "other words" median sits above its mean).
    for b, mean, med in zip(bars, means, medians):
        ax.text(b.get_x() + b.get_width() / 2, max(mean, med) + 0.022, f"{mean:.1%}",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="medium")

    # Gap arrow drawn in the gutter between the bars rather than on top of
    # one of them, so it can't cross a value label.
    gap = means[1] - means[0]
    ax.annotate("", xy=(0.5, means[0]), xytext=(0.5, means[1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(0.545, (means[0] + means[1]) / 2, f"{gap * 100:.1f} pts\nlower",
            fontsize=10, color=INK, fontweight="semibold", va="center")

    ax.set_ylim(0, 0.68)
    style(ax, "Mean per-class recall")
    ax.set_title("ASL signs articulated at the face are recognized measurably worse",
                 fontsize=12, color=INK, fontweight="semibold", loc="left", pad=30)
    ax.text(0, 1.02, f"Mann-Whitney U = {fh['mannwhitney_u']:.0f},  one-sided p = {fh['p_one_sided']:.4f}"
                      "   ·   groups fixed before any score was inspected",
            transform=ax.transAxes, fontsize=9, color=INK_3, va="bottom")
    save(fig, "fig_5_6_face_hypothesis.png")


# ------------------------------------------------------------------- 5.8
def benchmark_figure(asl_words_acc):
    # Published multi-stream skeletal+RGB results (Maruyama et al., 2021) and
    # this project's own measurement. Datasets differ, so this is contextual
    # rather than a controlled comparison -- stated in the caption and in §5.8.
    pub_x = [100, 300, 1000, 2000]
    pub_y = [0.8138, 0.7343, 0.6361, 0.4726]

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(pub_x, pub_y, color=BLUE, linewidth=2, marker="o", markersize=8,
            label="Published multi-stream (RGB + skeleton, server-class)", zorder=3)
    for x, y in zip(pub_x, pub_y):
        ax.annotate(f"{y:.1%}", xy=(x, y), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=9, color=INK_2)

    ax.plot([250], [asl_words_acc], marker="D", markersize=11, color=ORANGE,
            label="SignBridge ASL words (hand landmarks only, on-device)", zorder=4,
            linestyle="none")
    ax.annotate(f"{asl_words_acc:.1%}\n250 classes", xy=(250, asl_words_acc),
                xytext=(14, -26), textcoords="offset points", fontsize=9.5,
                color=ORANGE, fontweight="semibold")

    ax.set_xscale("log")
    ax.set_xticks([100, 250, 300, 1000, 2000])
    ax.set_xticklabels(["100", "250", "300", "1000", "2000"])
    ax.set_xlabel("Vocabulary size (classes, log scale)", color=INK_2)
    ax.set_ylim(0.35, 0.95)
    style(ax, "Top-1 accuracy")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_2)
    ax.set_title(f"Accuracy falls steeply with vocabulary size — context for reading {asl_words_acc:.0%}",
                 fontsize=12, color=INK, fontweight="semibold", loc="left", pad=30)
    ax.text(0, 1.02, "Different datasets — contextual comparison, not a controlled benchmark",
            transform=ax.transAxes, fontsize=9, color=INK_3, va="bottom")
    save(fig, "fig_5_8_benchmark_context.png")


# ------------------------------------------------------------------- 5.7
def anchor_figure(fh):
    """The position-anchor result, shown as a prediction and its test.

    The anchor was not added because it helped and then explained
    afterwards. The reasoning came first: normalizing every frame to the
    wrist is what lets a handshape be recognized wherever it is held, and
    is therefore also what erases the difference between signs
    distinguished only by *where* they are held. That predicts a specific,
    falsifiable thing -- signs articulated at the face should benefit more
    than the rest. Plotting the two groups side by side is what makes the
    prediction checkable rather than merely asserted, so this figure shows
    the split, not just the headline gain.
    """
    groups = ["Face-located words\n(prediction: gains most)", "All other words"]
    before = [0.4421, 0.5414]
    after = [fh["face_located"]["mean_recall"], fh["other_words"]["mean_recall"]]

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    x = range(len(groups))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], before, w, color=INK_3, zorder=3,
                label="Landmarks normalized only")
    b2 = ax.bar([i + w / 2 for i in x], after, w, color=AQUA, zorder=3,
                label="With position anchor")
    label_bars(ax, b1, before)
    label_bars(ax, b2, after)

    for i, (s, t) in enumerate(zip(before, after)):
        ax.annotate(f"{(t - s) * 100:+.2f} pts", xy=(i, max(s, t) + 0.075), ha="center",
                    fontsize=10, color=INK, fontweight="semibold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    # Headroom for the legend to sit clear of the "+x.xx pts" annotation
    # above the taller pair, which it otherwise overlaps.
    ax.set_ylim(0, 0.90)
    style(ax, "Mean per-class recall")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK_2)
    ax.set_title("The position anchor helped exactly the group it was predicted to",
                 fontsize=12, color=INK, fontweight="semibold", loc="left", pad=30)
    ax.text(0, 1.02,
            f"Gap narrowed 9.93 -> {(after[1] - after[0]) * 100:.2f} pts   ·   "
            f"one-sided p {0.0034:.4f} -> {fh['p_one_sided']:.4f}   ·   "
            "groups fixed before any score was inspected",
            transform=ax.transAxes, fontsize=9, color=INK_3, va="bottom")
    save(fig, "fig_5_7_position_anchor.png")


# ------------------------------------------------------------------- 5.9.3
def rotation_figure():
    data = load("asl_letter_rotation_comparison.json")
    angles = sorted((int(k) for k in data), key=int)
    old_vals = [data[str(a)][0] for a in angles]
    new_vals = [data[str(a)][1] for a in angles]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(angles, old_vals, linewidth=2, color=INK_3, marker="o", markersize=6,
            zorder=3, label="Old sequence model (shipped pre-fix)")
    ax.plot(angles, new_vals, linewidth=2, color=AQUA, marker="o", markersize=6,
            zorder=3, label="New direct classifier (shipped)")
    ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
    ax.axvspan(-15, 15, color=BLUE, alpha=0.05, zorder=0)
    ax.text(0, 0.99, "offline benchmark's\nentire operating range", ha="center", va="top",
            fontsize=8.3, color=INK_2)
    ax.annotate(f"{new_vals[0]:.0%}", xy=(angles[0], new_vals[0]), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9, color=INK, fontweight="semibold")
    ax.annotate(f"{old_vals[0]:.0%}", xy=(angles[0], old_vals[0]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=9, color=INK_3)
    ax.annotate(f"{new_vals[-1]:.0%}", xy=(angles[-1], new_vals[-1]), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9, color=INK, fontweight="semibold")
    ax.annotate(f"{old_vals[-1]:.0%}", xy=(angles[-1], old_vals[-1]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=9, color=INK_3)

    ax.set_xlabel("Simulated hand rotation (degrees)", color=INK_2)
    ax.legend(frameon=False, loc="lower center", fontsize=9.5, labelcolor=INK_2)
    style(ax, "Letter accuracy (held-out)")
    ax.set_title("The direct classifier wins at every rotation angle, including 0°",
                 fontsize=12, color=INK, fontweight="semibold", loc="left", pad=14)
    save(fig, "fig_5_9_rotation_robustness.png")


if __name__ == "__main__":
    print("Generating thesis figures...")
    bdsl = load("metrics_bdsl.json")
    asl = load("metrics_asl.json")
    fh = load("face_hypothesis.json")

    category_figure(bdsl, "BdSL: the 82.8% headline is carried by the word classes",
                    "fig_5_1_bdsl_by_category.png",
                    "All categories evaluated on 8 held-out signers")
    category_figure(asl, "ASL: words and static poses are not the same problem",
                    "fig_5_2_asl_by_category.png",
                    "Words: held-out signers  ·  Digits/Letters: random split (no signer identity available)")
    ablation_figure()
    defect_figure()
    confidence_figure()
    face_figure(fh)
    anchor_figure(fh)
    benchmark_figure(asl["by_category"]["word"]["accuracy"])
    rotation_figure()
    print(f"\nAll figures written to {os.path.relpath(OUT, HERE)}")
