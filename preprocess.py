"""
Pre-processing for the Perceived Stress Score project.

1. Data cleanup: drop rows where Perceived_Stress_Score is NaN.
2. Discretization of Age, Daily_Usage_Hours and Perceived_Stress_Score, two ways:
   a. Equal-width   - each category spans an equal slice of the value range
                      (PSS 0-40 -> 0-10, 10-20, 20-30, 30-40).
   b. Equal-frequency - each category holds (as close as possible) the same
                      number of students.
   Both data sets are written to CSV with the same columns as no_nan_data.csv,
   so either can be dropped into stratifydata.py.

Run:  .venv/bin/python preprocess.py
"""

import numpy as np
import pandas as pd

RAW_CSV = "Social_media_impact_on_life.csv"
EQUAL_WIDTH_CSV = "equal_width_data.csv"
EQUAL_FREQ_CSV = "equal_frequency_data.csv"

OUTPUT_COLUMNS = [
    "Student_ID", "Age", "Gender", "Academic_Level", "Primary_Platform",
    "Daily_Usage_Hours", "Device_Type", "Late_Night_Usage", "Perceived_Stress_Score",
]

AGE_LABELS_WIDTH = ["Teen", "College", "Graduate"]
LEVEL_LABELS = ["Low", "Medium", "High"]
STRESS_LABELS = ["Low", "Medium Low", "Medium High", "High"]


# --------------------------------------------------------------------------
# 1. Cleanup
# --------------------------------------------------------------------------
def load_clean(path=RAW_CSV):
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["Perceived_Stress_Score"]).reset_index(drop=True)
    print(f"Loaded {before} rows, dropped {before - len(df)} with NaN stress score -> {len(df)} rows")
    return df


# --------------------------------------------------------------------------
# 2a. Equal-width binning
# --------------------------------------------------------------------------
# Bins are left-closed [a, b); the last bin also includes its upper edge.
EQUAL_WIDTH_EDGES = {
    # Age groups from the spec: teen 15-17, college 18-22, grad 23-26.
    "Age": ([15, 18, 23, 27], AGE_LABELS_WIDTH),
    # Usage on a 0-15 hour scale (observed max is 14) -> 5-hour slices.
    "Daily_Usage_Hours": ([0, 5, 10, 15.0001], LEVEL_LABELS),
    # PSS is a 0-40 scale -> 10-point slices.
    "Perceived_Stress_Score": ([0, 10, 20, 30, 40.0001], STRESS_LABELS),
}


def equal_width(df):
    out = df.copy()
    for col, (edges, labels) in EQUAL_WIDTH_EDGES.items():
        out[col] = pd.cut(df[col], bins=edges, labels=labels, right=False)
        assert out[col].notna().all(), f"{col}: value outside equal-width edges"
    return out


# --------------------------------------------------------------------------
# 2b. Equal-frequency binning
# --------------------------------------------------------------------------
def equal_frequency_edges(values, k):
    """
    Choose k-1 cut points so the k bins are as close to n/k students each as
    possible. Students with the same value always land in the same bin (a
    19-year-old is never split across two age groups), so for heavily tied
    columns like Age the counts can only be approximately equal.

    Exhaustive search over cut positions between distinct values, minimising
    the largest deviation from n/k. Cheap: columns have <= ~150 distinct values.
    """
    uniq, counts = np.unique(values, return_counts=True)
    n, target = counts.sum(), counts.sum() / k
    cum = np.concatenate([[0], np.cumsum(counts)])  # cum[i] = students with value < uniq[i]

    best, best_cost = None, np.inf

    def search(start, cuts):
        nonlocal best, best_cost
        if len(cuts) == k - 1:
            sizes = np.diff([0] + [cum[c] for c in cuts] + [n])
            cost = np.abs(sizes - target).max()
            if cost < best_cost:
                best, best_cost = list(cuts), cost
            return
        for c in range(start, len(uniq) - (k - 1 - len(cuts)) + 1):
            # prune: once this bin is over target by more than the best cost, stop
            prev = cum[cuts[-1]] if cuts else 0
            if cum[c] - prev - target > best_cost:
                break
            search(c + 1, cuts + [c])

    search(1, [])
    # Edges sit on distinct values: bin i = [edge_i, edge_{i+1})
    return [uniq[0]] + [uniq[c] for c in best] + [np.nextafter(uniq[-1], np.inf)]


def equal_frequency(df):
    out = df.copy()
    edges_used = {}
    for col, labels in [("Age", LEVEL_LABELS),
                        ("Daily_Usage_Hours", LEVEL_LABELS),
                        ("Perceived_Stress_Score", STRESS_LABELS)]:
        edges = equal_frequency_edges(df[col].to_numpy(), len(labels))
        out[col] = pd.cut(df[col], bins=edges, labels=labels, right=False)
        assert out[col].notna().all()
        edges_used[col] = edges
    return out, edges_used


# --------------------------------------------------------------------------
def describe(name, binned, raw):
    print(f"\n=== {name} ===")
    for col in ["Age", "Daily_Usage_Hours", "Perceived_Stress_Score"]:
        print(f"  {col}")
        for label, grp in raw.groupby(binned[col], observed=False)[col]:
            rng = f"{grp.min():g}-{grp.max():g}" if len(grp) else "(empty)"
            print(f"    {label:<12} n={len(grp):>5}  ({len(grp) / len(raw):5.1%})  values {rng}")


def main():
    df = load_clean()

    width_df = equal_width(df)
    freq_df, _ = equal_frequency(df)

    width_df[OUTPUT_COLUMNS].to_csv(EQUAL_WIDTH_CSV, index=False)
    freq_df[OUTPUT_COLUMNS].to_csv(EQUAL_FREQ_CSV, index=False)

    describe("Equal-width bins", width_df, df)
    describe("Equal-frequency bins", freq_df, df)
    print(f"\nWrote {EQUAL_WIDTH_CSV} and {EQUAL_FREQ_CSV}")


if __name__ == "__main__":
    main()
