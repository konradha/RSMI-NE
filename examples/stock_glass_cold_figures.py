"""Draw the registered frozen-transfer results from the two released CSV tables."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


OUT = Path(__file__).resolve().parents[1] / "results" / "cold-frozen-transfer-320-340"
SEEDS = (6101, 6102, 6103)
SEED_STYLE = (("#007677", "o"), ("#ba501a", "s"), ("#65549c", "^"))
TARGETS = (3.20, 3.40)
PRIMARY = ((3.05, "source"), (3.20, "500"), (3.40, "500"))
LEVEL_LABELS = ("3.05\nsource TEST", "3.20\n500M", "3.40\n500M")
TARGET_LABELS = ("β = 3.20", "β = 3.40")


def _index(path, key):
    with path.open(newline="") as stream:
        return {key(row): row for row in csv.DictReader(stream)}


def _point(ax, x, row, style):
    """Draw the given endpoints, even if a percentile interval misses its estimate."""
    color, marker = style
    value, lo, hi = (float(row[k] or "nan") for k in ("value", "lo", "hi"))
    if math.isfinite(lo) and math.isfinite(hi):
        ax.vlines(x, lo, hi, color=color, linewidth=1.6, zorder=2)
        ax.plot([x, x], [lo, hi], linestyle="none", marker="_", color=color, ms=6)
    if math.isfinite(value):
        ax.plot(x, value, linestyle="none", marker=marker, color=color, ms=5, zorder=3)


def _categories(ax, labels):
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.48, len(labels) - 0.52)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.spines[["right", "top"]].set_visible(False)


def _by_seed(ax, labels, row_at):
    _categories(ax, labels)
    for j in range(len(labels)):
        for k, seed in enumerate(SEEDS):
            _point(ax, j + 0.16 * (k - 1), row_at(j, seed), SEED_STYLE[k])


def _legend(ax, styles, labels):
    handles = [Line2D([], [], linestyle="none", color=c, marker=m, markersize=5)
               for c, m in styles]
    ax.legend(handles, labels, loc="best", framealpha=0.94, fontsize=9)


def _seed_legend(ax):
    _legend(ax, SEED_STYLE, [f"Stock {seed}" for seed in SEEDS])


def _save(fig, out, name, caption):
    fig.supxlabel(caption, fontsize=10, color="#333333")
    fig.savefig(out / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw(out: Path):
    estimates = _index(out / "estimates.csv", lambda r: (
        float(r["beta"]), r["age"], r["view"], int(r["seed"]), r["metric"]))
    contrasts = _index(out / "contrasts.csv", lambda r: (
        r["name"], r["view"], int(r["seed"]), r["metric"]))

    def level(j, seed, metric):
        return estimates[*PRIMARY[j], "primary", seed, metric]

    # Keep each stock fit in its own column; colors here identify observables.
    scalar = (("T", "Frozen stock", ("#007677", "o")),
              ("linear_T", "TRAIN site-linear", ("#395387", "s")),
              ("vac_T", "Center vacancy", ("#856331", "^")))
    labels = (("learned_A", "Frozen labels", ("#007677", "o")),
              ("null_A", "Mean of 64 fixed nulls", ("#845375", "D")))
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharey="row", constrained_layout=True)
    for i, metrics in enumerate((scalar, labels)):
        for k, seed in enumerate(SEEDS):
            ax = axes[i, k]
            _categories(ax, LEVEL_LABELS)
            for j in range(len(PRIMARY)):
                for m, (metric, _, style) in enumerate(metrics):
                    x = j + 0.18 * (m - (len(metrics) - 1) / 2)
                    _point(ax, x, level(j, seed, metric), style)
            if i == 0:
                ax.set_title(f"Stock {seed}")
            else:
                ax.axhline(1, color="0.45", linewidth=1, linestyle=":")
        _legend(axes[i, 0], [v[2] for v in metrics], [v[1] for v in metrics])
    axes[0, 0].set_ylabel("Scalar transfer T")
    axes[1, 0].set_ylabel("Normalized label agreement A")
    fig.suptitle("Does frozen structure transfer to colder parents?")
    _save(fig, out, "cold-transfer.png",
          "First eligible star per parent. Whiskers: pointwise 95% parent-bootstrap intervals.\n"
          "Agreement panel uses its own scale; dotted line is saturation (A = 1).")

    fig, (ax, ess) = plt.subplots(1, 2, figsize=(11.7, 4.9), constrained_layout=True)

    def practical(j, seed):
        if j == 2:
            return contrasts["temperature500", "primary", seed, "advantage"]
        return level(j + 1, seed, "advantage")

    _by_seed(ax, ("3.20\nD", "3.40\nD", "3.40 − 3.20\nΔD"), practical)
    ax.axhline(0, color="0.45", linewidth=0.9)
    ax.axhline(0.01, color="#884220", linewidth=1.4, linestyle="--")
    ax.set(title="Specificity at 500M and its temperature contrast",
           ylabel="Agreement difference (dashed: +0.01 margin)")
    _seed_legend(ax)
    _by_seed(ess, TARGET_LABELS, lambda j, s: level(j + 1, s, "deficit_ess"))
    ess.axhline(10, color="#884220", linewidth=1.4, linestyle="--")
    ess.set(title="How many parents carry the residual deficit?",
            ylabel="Deficit effective parent count (dashed: 10)")
    fig.suptitle("Learned minus mean-null agreement (D): practical effect and rarity")
    _save(fig, out, "cold-specificity.png",
          "Whiskers: pointwise 95% parent-bootstrap intervals, conditional on frozen fits and nulls.\n"
          "Intervals alone cannot rule out rare events: population claims also require adequate denominators and deficit support for both learned and null labels.")

    diagnostics = (
        ("MI", "Boundary–label mutual information", "I(B; label) [bits]"),
        ("H", "Marginal label entropy", "H(label) [bits]"),
        ("Hcond", "Mean conditional label entropy", "H(label | B) [bits]"),
        ("state_same", "Whole-star collision", "Probability of identical states"),
        ("within_var", "Mean within-boundary variance", "Within variance [stock units²]"),
        ("total_var", "Total scalar variance", "Total variance [stock units²]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.5), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, diagnostics):
        _by_seed(ax, LEVEL_LABELS, lambda j, s: level(j, s, metric))
        ax.set(title=title, ylabel=ylabel)
        if metric == "state_same":
            ax.axhline(1, color="0.45", linewidth=1, linestyle=":")
    _seed_legend(axes[0, 0])
    fig.suptitle("What remains uncertain at a frozen boundary?")
    _save(fig, out, "cold-information.png",
          "First eligible stars; pointwise 95% parent intervals. Panels have separate vertical scales.\n"
          "Collision uses two independent conditional Gibbs draws at the same boundary, not two trajectory times.")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.1), constrained_layout=True)
    for i, (metric, label) in enumerate((("T", "Scalar transfer T"), ("advantage", "Specificity D"))):
        paired, levels = axes[i]
        _by_seed(paired, TARGET_LABELS, lambda j, s: contrasts[
            ("age320", "age340")[j], "primary", s, metric])
        paired.axhline(0, color="0.45", linewidth=1)
        paired.set_ylabel(f"{label}: 500M − 300M")
        _by_seed(levels, ("3.20\nfirst", "3.20\nall", "3.40\nfirst", "3.40\nall"),
                 lambda j, s: estimates[TARGETS[j // 2], "500", ("primary", "all")[j % 2], s, metric])
        levels.set_ylabel(f"{label} at 500M")
    axes[0, 0].set_title("Age change · complete parent pairs, first star")
    axes[0, 1].set_title("Anchor sensitivity · first vs. all eligible stars")
    _seed_legend(axes[0, 0])
    fig.suptitle("Sensitivity to preparation age and anchor choice")
    _save(fig, out, "cold-sensitivity.png",
          "Left: bootstrap intervals for paired changes. Right: intervals for levels, not for their differences.\n"
          "Parents, not stars, are the sampling units in both views. No source age comparison is defined.")

    fig, axs = plt.subplots(2, 3, figsize=(12.4, 7), constrained_layout=True)
    for k, seed in enumerate(SEEDS):
        ax = axs[0, k]
        _categories(ax, LEVEL_LABELS)
        for j in range(3):
            for i, (v, style) in enumerate((("packing_bits", ("#845375","D")), ("directional_bits", ("#007677","o")))):
                _point(ax, j+.16*(i-.5), level(j,seed,v),style)
        ax.set(title=f"Frozen stock {seed}", ylabel="Expected forecast loss [bits / star]")
        if k==0: _legend(ax, [("#845375","D"),("#007677","o")], ["Frozen packing head","Frozen directional head"])
        ax=axs[1,k]; _categories(ax, ["00","01","10","11"])
        for h in range(4):
            for j in range(3): _point(ax,h+.17*(j-1),level(j,seed,f"gap_h{h}"),SEED_STYLE[j])
        ax.axhline(0,color="gray",ls=":",lw=1); ax.set(ylabel="Observed − Gibbs-predicted label fraction")
        if k==0: _legend(ax,SEED_STYLE,["β=3.05","β=3.20","β=3.40"])
    fig.suptitle("How well do unchanged source forecasts describe the colder boundaries?")
    _save(fig,out,"cold-forecasts.png","One observed interior per parent; the Gibbs probabilities are from enumeration, not additional observations.\nTop: 07 classifiers use the exterior only. Bottom: labels are defined separately for each frozen seed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT, help="Directory containing the two CSV files")
    draw(parser.parse_args().out)
