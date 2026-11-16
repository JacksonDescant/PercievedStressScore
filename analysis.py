"""
Exploratory analysis + figures for the Perceived Stress Score project.

Reads the raw CSV (NaN stress rows dropped) and the two discretized CSVs
written by preprocess.py, prints summary stats, and saves PNGs to figures/.

Run:  .venv/bin/python preprocess.py && .venv/bin/python analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from preprocess import (EQUAL_FREQ_CSV, EQUAL_WIDTH_CSV, LEVEL_LABELS,
                        AGE_LABELS_WIDTH, STRESS_LABELS, equal_frequency, load_clean)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ---- palette (validated categorical order + single-hue ordinal ramp) -------
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STRESS_RAMP = ["#b7d3f6", "#6da7ec", "#256abf", "#104281"]  # Low -> High
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#256abf", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10, "legend.frameon": False,
})

NUMERIC = ["Age", "Daily_Usage_Hours", "Weekend_Extra_Hours", "Sleep_Duration_Hours",
           "Sleep_Quality_Score", "Perceived_Stress_Score", "Mental_Health_Index",
           "Academic_Performance_GPA"]
CATEGORICAL = ["Gender", "Academic_Level", "Primary_Platform", "Device_Type",
               "Late_Night_Usage", "Social_Comparison_Frequency"]


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def ordered(series, order):
    return pd.Categorical(series, categories=order, ordered=True)


# ---------------------------------------------------------------------------
def fig_raw_distributions(df):
    """Histograms of the three variables we discretize, with both sets of cut lines."""
    _, freq_edges = equal_frequency(df)
    width_cuts = {"Age": [18, 23], "Daily_Usage_Hours": [5, 10],
                  "Perceived_Stress_Score": [10, 20, 30]}
    specs = [("Age", np.arange(14.5, 27.5, 1), "Age (years)"),
             ("Daily_Usage_Hours", np.arange(0, 14.6, 0.5), "Daily usage (hours)"),
             ("Perceived_Stress_Score", np.arange(-0.5, 41.5, 1), "Perceived Stress Score (0-40)")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, bins, xlabel) in zip(axes, specs):
        ax.hist(df[col], bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
        ymax = ax.get_ylim()[1]
        for x in width_cuts[col]:
            ax.axvline(x - (0.5 if col != "Daily_Usage_Hours" else 0), color=ORANGE, lw=2)
        for x in freq_edges[col][1:-1]:
            ax.axvline(x - (0.5 if col != "Daily_Usage_Hours" else 0.05), color=INK, lw=1.5, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Students")
        ax.set_title(col.replace("_", " "))
        ax.set_ylim(0, ymax * 1.05)
    axes[0].plot([], [], color=ORANGE, lw=2, label="Equal-width cut")
    axes[0].plot([], [], color=INK, lw=1.5, ls="--", label="Equal-frequency cut")
    axes[0].legend(loc="upper right")
    fig.suptitle("Where each binning strategy cuts the raw distributions",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "01_raw_distributions_with_cuts.png")


def fig_bin_balance(width, freq):
    """Students per category: equal-width vs equal-frequency."""
    specs = [("Age", AGE_LABELS_WIDTH, LEVEL_LABELS),
             ("Daily_Usage_Hours", LEVEL_LABELS, LEVEL_LABELS),
             ("Perceived_Stress_Score", STRESS_LABELS, STRESS_LABELS)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (col, w_order, f_order) in zip(axes, specs):
        w = width[col].value_counts().reindex(w_order)
        f = freq[col].value_counts().reindex(f_order)
        x = np.arange(len(w_order))
        bw = 0.38
        b1 = ax.bar(x - bw / 2 - 0.01, w.values, bw, color=ORANGE, label="Equal-width")
        b2 = ax.bar(x + bw / 2 + 0.01, f.values, bw, color=BLUE, label="Equal-frequency")
        ax.bar_label(b1, padding=2, fontsize=8, color=INK_2)
        ax.bar_label(b2, padding=2, fontsize=8, color=INK_2)
        ax.axhline(len(width) / len(w_order), color=INK_2, lw=1, ls=":")
        labels = ([f"{a}\n{b}" for a, b in zip(w_order, f_order)] if w_order != f_order else w_order)
        ax.set_xticks(x, labels)
        ax.set_title(col.replace("_", " "))
        ax.set_ylabel("Students")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    fig.suptitle("Category sizes under each strategy (dotted line = perfectly even split)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "02_category_balance.png")


def fig_usage_vs_stress(df):
    r = df["Daily_Usage_Hours"].corr(df["Perceived_Stress_Score"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(df))
    ax.scatter(df["Daily_Usage_Hours"], df["Perceived_Stress_Score"] + jitter,
               s=10, alpha=0.35, color=BLUE, edgecolors="none")
    m, b = np.polyfit(df["Daily_Usage_Hours"], df["Perceived_Stress_Score"], 1)
    xs = np.linspace(df["Daily_Usage_Hours"].min(), df["Daily_Usage_Hours"].max(), 50)
    ax.plot(xs, m * xs + b, color=ORANGE, lw=2.5, label=f"Linear fit (r = {r:.2f})")
    ax.set_xlabel("Daily usage (hours)")
    ax.set_ylabel("Perceived Stress Score")
    ax.set_title("More daily social-media use goes with higher stress")
    ax.legend(loc="upper left")
    save(fig, "03_usage_vs_stress_scatter.png")


def fig_correlation(df):
    corr = df[NUMERIC].corr()
    names = [c.replace("_", " ").replace("Academic Performance ", "") for c in NUMERIC]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERIC)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(NUMERIC)), names)
    ax.grid(False)
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation between numeric columns")
    save(fig, "04_correlation_heatmap.png")


def fig_stress_by_category(df):
    """Mean PSS (with 95% CI) for each level of the categorical features."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    overall = df["Perceived_Stress_Score"].mean()
    for ax, col in zip(axes.flat, CATEGORICAL):
        g = df.groupby(col)["Perceived_Stress_Score"].agg(["mean", "std", "count"])
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
        g = g.sort_values("mean")
        y = np.arange(len(g))
        ax.errorbar(g["mean"], y, xerr=g["ci"], fmt="o", ms=8, color=BLUE,
                    ecolor=BLUE, elinewidth=2, capsize=0)
        ax.axvline(overall, color=INK_2, lw=1, ls=":")
        ax.set_yticks(y, [f"{i}  (n={n})" for i, n in zip(g.index.astype(str), g["count"])])
        ax.set_title(col.replace("_", " "))
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Mean Perceived Stress Score (95% CI)")
    fig.suptitle(f"Average stress by group (dotted line = overall mean {overall:.1f})",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "05_stress_by_category.png")


def fig_stress_mix(binned, tag, age_order, title_suffix):
    """Share of each stress category within each feature level (100% stacked bars)."""
    features = [("Age", age_order), ("Daily_Usage_Hours", LEVEL_LABELS),
                ("Late_Night_Usage", [False, True]), ("Academic_Level", ["High School", "Undergraduate", "Postgraduate"])]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, (col, order) in zip(axes, features):
        tab = pd.crosstab(binned[col], binned["Perceived_Stress_Score"], normalize="index")
        tab = tab.reindex(columns=STRESS_LABELS, fill_value=0)
        if order is not None:
            tab = tab.reindex(order)
        x = np.arange(len(tab))
        bottom = np.zeros(len(tab))
        for label, color in zip(STRESS_LABELS, STRESS_RAMP):
            vals = tab[label].to_numpy()
            ax.bar(x, vals, 0.7, bottom=bottom, color=color, edgecolor=SURFACE,
                   linewidth=2, label=label)
            bottom += vals
        ax.set_xticks(x, [str(i) for i in tab.index])
        ax.set_title(col.replace("_", " "))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of students")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title="Stress category", loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"Stress-category mix by feature - {title_suffix}",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, f"06_stress_mix_{tag}.png")


DENSITY_FEATURES = [
    ("Daily_Usage_Hours", np.arange(0.75, 14.5, 0.5), "Daily usage (hours)"),
    ("Sleep_Duration_Hours", np.arange(2.95, 10.6, 0.3), "Sleep duration (hours)"),
    ("Mental_Health_Index", np.arange(31.5, 99.5, 2), "Mental Health Index"),
    ("Academic_Performance_GPA", np.arange(1.85, 4.06, 0.1), "GPA"),
]


def _density_overlay(df, group_col, groups, colors, legend_title, title, footnote, name,
                     features=DENSITY_FEATURES):
    """Overlaid density histograms, one panel per numeric feature, one color per group."""
    fig, axes = plt.subplots(1, len(features), figsize=(4.6 * len(features), 4.2))
    for ax, (col, bins, xlabel) in zip(axes, features):
        for (label, value), color in zip(groups, colors):
            vals = df.loc[df[group_col] == value, col].dropna()
            ax.hist(vals, bins=bins, density=True, alpha=0.55, color=color,
                    edgecolor=SURFACE, linewidth=0.6, label=f"{label} (n={len(vals)})")
        ax.set_title(xlabel)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Density")
    fig.suptitle(title, x=0.01, y=1.06, ha="left", fontsize=14, fontweight="bold", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=legend_title, loc="lower right", ncol=len(groups),
               bbox_to_anchor=(0.995, 0.97), alignment="left")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=9, color=INK_2)
    fig.tight_layout()
    save(fig, name)


def fig_density_by_late_night(df):
    features = [("Perceived_Stress_Score", np.arange(-0.5, 41, 1), "Perceived Stress Score")]
    features += DENSITY_FEATURES[:3]
    _density_overlay(
        df, "Late_Night_Usage", [("No", False), ("Yes", True)], [BLUE, ORANGE],
        "Late-night use", "Distributions by late-night social-media use",
        "Histograms show density, so groups of different size are comparable.",
        "07_density_by_late_night.png", features)


def fig_density_by_stress(df, freq):
    lo = freq["Perceived_Stress_Score"] == "Low"
    hi = freq["Perceived_Stress_Score"] == "High"
    lo_rng = df.loc[lo, "Perceived_Stress_Score"]
    hi_rng = df.loc[hi, "Perceived_Stress_Score"]
    tagged = df.assign(_grp=np.where(lo, "low", np.where(hi, "high", "mid")))
    _density_overlay(
        tagged, "_grp",
        [(f"Low stress (PSS {lo_rng.min():g}-{lo_rng.max():g})", "low"),
         (f"High stress (PSS {hi_rng.min():g}-{hi_rng.max():g})", "high")],
        [BLUE, ORANGE], "Stress group",
        "Lowest vs highest stress quarter of students",
        "Histograms show density. Groups are the Low and High equal-frequency stress bins "
        "(about a quarter of students each).",
        "08_density_low_vs_high_stress.png")


def fig_category_counts(df):
    """Horizontal bar charts of student counts per category, largest first."""
    cols = ["Primary_Platform", "Academic_Level", "Gender", "Device_Type",
            "Social_Comparison_Frequency", "Late_Night_Usage"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    natural_order = {  # ordinal columns keep their scale order (top to bottom)
        "Social_Comparison_Frequency": ["Never", "Rarely", "Sometimes", "Frequently", "Always"],
        "Late_Night_Usage": ["False", "True"],
    }
    for ax, col in zip(axes.flat, cols):
        counts = df[col].astype(str).value_counts()
        if col in natural_order:
            counts = counts.reindex(natural_order[col][::-1])
        else:
            counts = counts.sort_values()
        if col == "Late_Night_Usage":
            counts.index = ["Yes", "No"]
        bars = ax.barh(counts.index, counts.values, height=0.7, color=BLUE)
        ax.bar_label(bars, padding=3, fontsize=8, color=INK_2)
        ax.set_xlim(0, counts.max() * 1.15)
        ax.set_title(col.replace("_", " "))
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Students")
    fig.suptitle("How many students fall in each category",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "09_category_counts.png")


def print_summary(df, width, freq):
    print("\n--- Numeric summary (cleaned data) ---")
    print(df[NUMERIC].describe().round(2).T.to_string())
    print("\n--- Correlation with Perceived_Stress_Score ---")
    print(df[NUMERIC].corr()["Perceived_Stress_Score"].drop("Perceived_Stress_Score")
          .sort_values(key=abs, ascending=False).round(3).to_string())
    for name, b in [("equal-width", width), ("equal-frequency", freq)]:
        print(f"\n--- Mean raw PSS by usage category ({name}) ---")
        print(df.groupby(b["Daily_Usage_Hours"])["Perceived_Stress_Score"]
              .agg(["count", "mean"]).reindex(LEVEL_LABELS).round(2).to_string())


def main():
    df = load_clean()
    width = pd.read_csv(EQUAL_WIDTH_CSV)
    freq = pd.read_csv(EQUAL_FREQ_CSV)
    assert (width["Student_ID"] == df["Student_ID"]).all()
    assert (freq["Student_ID"] == df["Student_ID"]).all()

    print_summary(df, width, freq)

    print("\nFigures:")
    fig_raw_distributions(df)
    fig_bin_balance(width, freq)
    fig_usage_vs_stress(df)
    fig_correlation(df)
    fig_stress_by_category(df)
    width_extra = width.assign(Academic_Level=df["Academic_Level"])
    freq_extra = freq.assign(Academic_Level=df["Academic_Level"])
    fig_stress_mix(width_extra, "equal_width", AGE_LABELS_WIDTH, "equal-width bins")
    fig_stress_mix(freq_extra, "equal_frequency", LEVEL_LABELS, "equal-frequency bins")
    fig_density_by_late_night(df)
    fig_density_by_stress(df, freq)
    fig_category_counts(df)


if __name__ == "__main__":
    main()
