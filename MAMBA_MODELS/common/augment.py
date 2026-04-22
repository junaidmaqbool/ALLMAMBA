"""
augment.py -- Smart bin-based oversampling for HB datasets.
============================================================
Clinical HB data follows a bell curve centred near the mean (~11-13 g/dL).
The model therefore sees far more "average" samples than peripheral values
(severely anaemic <9 or borderline normal ~12-13), causing it to learn
"predict the mean" and plateau.

This module fixes that by:
  1. Filtering each dataset to [HB_FILTER_MIN, HB_FILTER_MAX] (removes outliers)
  2. Merging multiple datasets (left_eye + right_eye, etc.) when provided
  3. Dividing the HB range into equal-width bins
  4. Giving each sample a weight = 1 / count(its bin)
  5. Using WeightedRandomSampler so every bin contributes equally per epoch

Result: peripherally anaemic samples (HB~7-9) appear as often per epoch
as samples near the threshold (~12), breaking the mean-prediction plateau.

TRAIN only -- validation always uses real, unmodified, unweighted samples.

Usage
-----
    from common.augment import (
        merge_and_filter_datasets,
        plot_hb_distribution,
        make_balanced_loader,
    )
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset merging + HB range filter
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_filter_datasets(
    dataframes: list,          # list of pd.DataFrame (one per dataset slot)
    image_dirs: list,          # list of image directory strings (matching dataframes)
    hb_col:     str,
    image_col:  str,
    hb_filter_min: float,
    hb_filter_max: float,
    find_image_path_fn,        # callable(pid, image_dir) -> path | None
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Merge multiple dataset DataFrames, attach image paths,
    and filter to [hb_filter_min, hb_filter_max].

    Returns a single clean DataFrame with columns:
        HB_COL, IMAGE_COL, '_img_path', '_source_dir'
    """
    import pandas as pd

    merged_parts = []
    for i, (df, img_dir) in enumerate(zip(dataframes, image_dirs)):
        if df is None or img_dir == "":
            continue
        df = df.copy()
        df["_source_dir"] = img_dir

        # Validate HB column
        df[hb_col] = pd.to_numeric(df[hb_col], errors="coerce")
        before = len(df)
        df = df[df[hb_col].notna()].copy()
        if verbose and (before - len(df)) > 0:
            print(f"  [Dataset {i+1}] Dropped {before - len(df)} rows with non-numeric HB.")

        # Attach image paths
        def _find(row):
            return find_image_path_fn(str(row[image_col]), row["_source_dir"])
        df["_img_path"] = df.apply(_find, axis=1)
        missing = df["_img_path"].isna().sum()
        if missing > 0:
            ex = df.loc[df["_img_path"].isna(), image_col].tolist()[:4]
            if verbose:
                print(f"  [Dataset {i+1}] Skipped {missing} rows — no image found. "
                      f"Examples: {ex}")
        df = df[df["_img_path"].notna()].copy()

        merged_parts.append(df)
        if verbose:
            print(f"  [Dataset {i+1}] {img_dir} -> {len(df)} valid samples")

    if not merged_parts:
        raise RuntimeError("No valid samples found across all dataset slots.")

    merged = pd.concat(merged_parts, ignore_index=True)
    before_filter = len(merged)
    merged = merged[
        (merged[hb_col] >= hb_filter_min) &
        (merged[hb_col] <= hb_filter_max)
    ].copy()
    removed = before_filter - len(merged)
    if verbose:
        print(f"\n  HB filter [{hb_filter_min}, {hb_filter_max}] g/dL: "
              f"kept {len(merged)} / {before_filter}  "
              f"({removed} removed as outliers)")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Distribution analysis + plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_hb_distribution(
    df,
    hb_col:        str,
    hb_thresh:     float,
    hb_filter_min: float,
    hb_filter_max: float,
    aug_bins:      int  = 10,
    save_path:     str  = "hb_distribution.png",
):
    """
    Plot 4-panel HB distribution analysis and print statistics.
    Run this ONCE before training to understand your data and decide
    whether USE_AUGMENTATION is needed.

    Panels:
      1. Raw HB histogram + KDE + filter range overlay
      2. Bin counts after filtering (underrepresented bins highlighted)
      3. Class balance (Anemic vs Normal)
      4. Oversampling weights per bin (preview of what augmentation does)
    """
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import gaussian_kde
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    hb  = df[hb_col].dropna()
    hb_f = hb[(hb >= hb_filter_min) & (hb <= hb_filter_max)]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("HB Distribution Analysis  (run once before training)", fontsize=12)

    # ── Panel 1: Histogram + KDE ─────────────────────────────────────────
    ax = axes[0]
    bins_hist = np.linspace(hb.min() - 0.5, hb.max() + 0.5, 35)
    ax.hist(hb,   bins=bins_hist, density=True, alpha=0.35, color="steelblue",
            label="All data")
    ax.hist(hb_f, bins=bins_hist, density=True, alpha=0.45, color="orange",
            label=f"Filtered [{hb_filter_min},{hb_filter_max}]")
    if HAS_SCIPY and len(hb) > 1:
        x = np.linspace(hb.min(), hb.max(), 300)
        ax.plot(x, gaussian_kde(hb)(x),   "b-",          lw=1.5, label="KDE all")
        ax.plot(x, gaussian_kde(hb_f)(x), "darkorange",  lw=1.5,
                linestyle="--", label="KDE filtered")
    ax.axvline(hb_thresh,    color="red",   linestyle="--", lw=1.5,
               label=f"Threshold ({hb_thresh})")
    ax.axvline(hb.mean(),    color="green", linestyle=":",  lw=1.5,
               label=f"Mean ({hb.mean():.1f})")
    ax.axvspan(hb_filter_min, hb_filter_max, alpha=0.07, color="limegreen")
    ax.set_xlabel("HB (g/dL)"); ax.set_title("Distribution")
    ax.legend(fontsize=7)

    # ── Panel 2: Bin counts (filtered range) ─────────────────────────────
    ax = axes[1]
    bin_edges  = np.linspace(hb_filter_min, hb_filter_max, aug_bins + 1)
    counts, _  = np.histogram(hb_f, bins=bin_edges)
    centres    = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_count = counts.mean()
    bar_colors = ["salmon" if c < mean_count * 0.7 else "steelblue" for c in counts]
    ax.bar(centres, counts, width=(bin_edges[1] - bin_edges[0]) * 0.85,
           color=bar_colors, edgecolor="white")
    ax.axhline(mean_count, color="red", linestyle="--", lw=1.5,
               label=f"Mean ({mean_count:.0f})")
    ax.axvline(hb_thresh, color="orange", linestyle="--", lw=1.2,
               label=f"Threshold ({hb_thresh})")
    ax.set_xlabel("HB (g/dL)"); ax.set_ylabel("Count")
    ax.set_title(f"Sample counts per bin\n(red = underrepresented)")
    ax.legend(fontsize=8)

    # ── Panel 3: Class balance ────────────────────────────────────────────
    ax = axes[2]
    n_anemic = int((hb_f < hb_thresh).sum())
    n_normal = int((hb_f >= hb_thresh).sum())
    total_f  = n_anemic + n_normal
    bars = ax.bar(
        [f"Anemic\n(<{hb_thresh:.0f})", f"Normal\n(>={hb_thresh:.0f})"],
        [n_anemic, n_normal],
        color=["salmon", "steelblue"],
        edgecolor="white",
    )
    for bar, v in zip(bars, [n_anemic, n_normal]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total_f * 0.01,
                f"{v}\n({v/total_f*100:.1f}%)", ha="center", fontsize=10)
    ax.set_title("Class Balance\n(after filtering)")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(n_anemic, n_normal) * 1.25)

    # ── Panel 4: Oversampling weights preview ─────────────────────────────
    ax = axes[3]
    weights = compute_sample_weights(
        hb_f.values, n_bins=aug_bins,
        hb_min=hb_filter_min, hb_max=hb_filter_max,
    )
    # Average weight per bin
    bin_idx    = np.digitize(hb_f.values,
                              np.linspace(hb_filter_min, hb_filter_max + 1e-6,
                                          aug_bins + 1)) - 1
    bin_idx    = np.clip(bin_idx, 0, aug_bins - 1)
    avg_w      = [weights[bin_idx == b].mean() if (bin_idx == b).any() else 0
                  for b in range(aug_bins)]
    ax.bar(centres, avg_w, width=(bin_edges[1] - bin_edges[0]) * 0.85,
           color="mediumseagreen", edgecolor="white", alpha=0.8)
    ax.axvline(hb_thresh, color="orange", linestyle="--", lw=1.2,
               label=f"Threshold ({hb_thresh})")
    ax.set_xlabel("HB (g/dL)"); ax.set_ylabel("Avg oversampling weight")
    ax.set_title("Oversampling weights\n(what USE_AUGMENTATION does)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"  Chart saved: {save_path}")

    # ── Console statistics ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  HB DISTRIBUTION STATISTICS")
    print("=" * 55)
    print(f"  Total rows in CSV(s)     : {len(hb)}")
    print(f"  After filter             : {len(hb_f)}"
          f"  ({len(hb_f)/len(hb)*100:.1f}% kept,"
          f" {len(hb)-len(hb_f)} removed as outliers)")
    print(f"  HB mean  (filtered)      : {hb_f.mean():.2f} g/dL")
    print(f"  HB std   (filtered)      : {hb_f.std():.2f} g/dL")
    print(f"  HB median(filtered)      : {hb_f.median():.2f} g/dL")
    print(f"  HB range (filtered)      : [{hb_f.min():.2f}, {hb_f.max():.2f}] g/dL")
    print(f"  Anemic   (<{hb_thresh:.0f} g/dL)    : {n_anemic}"
          f"  ({n_anemic/total_f*100:.1f}%)")
    print(f"  Normal   (>={hb_thresh:.0f} g/dL)   : {n_normal}"
          f"  ({n_normal/total_f*100:.1f}%)")
    imbalance = abs(n_anemic - n_normal) / total_f
    if imbalance > 0.20:
        print(f"  *** Class imbalance = {imbalance*100:.0f}%"
              " -- USE_AUGMENTATION=True + FocalLoss recommended ***")
    print(f"\n  Bin breakdown ({aug_bins} equal bins, {hb_filter_min}-{hb_filter_max} g/dL):")
    for lo, hi, cnt in zip(bin_edges[:-1], bin_edges[1:], counts):
        flag = "  <-- underrepresented" if cnt < mean_count * 0.7 else ""
        print(f"    [{lo:.1f}, {hi:.1f}): {cnt:5d} samples{flag}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Compute per-sample weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(
    hb_values,
    n_bins:  int   = 10,
    hb_min:  float = None,
    hb_max:  float = None,
) -> np.ndarray:
    """
    Compute per-sample oversampling weights based on bin membership.
    Samples in sparse bins (peripheral HB) get higher weight so they
    appear more often per epoch than the central bell-curve mass.

    Weight = 1 / count(sample's bin)
    Result is scaled so the array sums to len(hb_values).

    Parameters
    ----------
    hb_values : array-like of HB values (already filtered)
    n_bins    : number of equal-width bins
    hb_min / hb_max : bin boundaries (defaults to data min/max)

    Returns
    -------
    np.ndarray of shape (N,) — one weight per sample
    """
    hb_arr = np.asarray(hb_values, dtype=float)
    lo     = hb_min if hb_min is not None else hb_arr.min()
    hi     = hb_max if hb_max is not None else hb_arr.max()

    bin_edges = np.linspace(lo, hi + 1e-6, n_bins + 1)   # +eps so max included
    bin_idx   = np.digitize(hb_arr, bin_edges) - 1        # 0-indexed
    bin_idx   = np.clip(bin_idx, 0, n_bins - 1)

    counts       = np.bincount(bin_idx, minlength=n_bins).astype(float)
    counts       = np.maximum(counts, 1)                  # avoid /0
    bin_weight   = 1.0 / counts                           # sparse bins → high weight
    sample_w     = bin_weight[bin_idx]
    return sample_w / sample_w.sum() * len(hb_arr)        # normalise: sum == N


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Balanced DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def make_balanced_loader(
    dataset,
    hb_values,
    batch_size:     int,
    num_workers:    int,
    device:         str,
    hb_filter_min:  float,
    hb_filter_max:  float,
    aug_bins:       int = 10,
) -> DataLoader:
    """
    Build a DataLoader that oversamples peripheral HB bins so each bin
    contributes equally to every training epoch.

    Parameters
    ----------
    dataset       : EyeHBDataset (already filtered to [hb_filter_min, hb_filter_max])
    hb_values     : list/array of raw HB values for each sample in dataset
    batch_size, num_workers, device : standard DataLoader params
    hb_filter_min / hb_filter_max   : used to define bin boundaries
    aug_bins      : number of equal-width bins (default 10)

    Returns
    -------
    DataLoader with WeightedRandomSampler (replaces shuffle=True)
    """
    weights = compute_sample_weights(
        hb_values,
        n_bins=aug_bins,
        hb_min=hb_filter_min,
        hb_max=hb_filter_max,
    )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),   # one full epoch worth of samples
        replacement=True,           # allows peripheral samples to repeat
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,            # sampler is mutually exclusive with shuffle
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
