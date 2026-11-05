"""
Generates every figure and every number quoted in docs/projects/eda/index.md.

    python code/eda.py

Figures go to docs/projects/eda/figures/, numbers to docs/projects/eda/results.json
(so the text of the report can be checked against the code).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (  # noqa: E402
    CATEGORICAL_FEATURES, DATA_PATH, HEAVY_TAIL_FEATURES,
    MIN_NEIGHBOURHOOD_FREQ, NUMERIC_FEATURES, RANDOM_STATE, TARGET,
    build_preprocessor, load_data, split_data,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "projects" / "eda" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = {}

# ---- style -----------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
# Okabe-Ito, fixed order so the same category always gets the same colour
CAT = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
ROOM_ORDER = ["Entire home/apt", "Private room", "Shared room"]
GROUP_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
ROOM_COL = dict(zip(ROOM_ORDER, CAT[:3]))
GROUP_COL = dict(zip(GROUP_ORDER, CAT[:5]))
SEQ = "viridis"
DPI = 140
PRICE_TICKS = [40, 100, 250, 600]
VMIN, VMAX = 3.5, 6.5  # log1p(price) colour range shared by every projection


def save(fig, n, slug, tight=True):
    path = FIG_DIR / f"fig{n:02d}_{slug}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.name}")


def r(x, nd=3):
    return None if x is None else float(np.round(x, nd))


def price_colorbar(mappable, ax):
    cb = plt.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("price")
    cb.set_ticks([np.log1p(v) for v in PRICE_TICKS])
    cb.set_ticklabels([f"${v}" for v in PRICE_TICKS])
    return cb


# ============================================================================
# 1. Initial inspection (on the raw file)
# ============================================================================
raw = pd.read_csv(DATA_PATH)
RESULTS["raw_shape"] = list(raw.shape)

missing = raw.isna().sum()
RESULTS["missing"] = {
    c: {"count": int(missing[c]), "pct": r(100 * missing[c] / len(raw), 2)}
    for c in raw.columns if missing[c] > 0
}
RESULTS["dup_rows"] = int(raw.duplicated().sum())
RESULTS["dup_ids"] = int(raw["id"].duplicated().sum())
RESULTS["rpm_nan_iff_no_reviews"] = bool(
    raw.loc[raw.number_of_reviews == 0, "reviews_per_month"].isna().all()
    and raw.loc[raw.number_of_reviews > 0, "reviews_per_month"].notna().all()
)
RESULTS["price_zero"] = int((raw.price == 0).sum())
RESULTS["price_ge_1000"] = int((raw.price >= 1000).sum())
RESULTS["price_max"] = int(raw.price.max())
RESULTS["min_nights_gt_365"] = int((raw.minimum_nights > 365).sum())
RESULTS["min_nights_gt_30"] = int((raw.minimum_nights > 30).sum())
RESULTS["min_nights_max"] = int(raw.minimum_nights.max())
RESULTS["avail_zero"] = int((raw.availability_365 == 0).sum())
RESULTS["avail_zero_pct"] = r(100 * (raw.availability_365 == 0).mean(), 1)
RESULTS["avail_365"] = int((raw.availability_365 == 365).sum())
RESULTS["rpm_gt_20"] = int((raw.reviews_per_month > 20).sum())
RESULTS["rpm_max"] = float(raw.reviews_per_month.max())
RESULTS["lat_range"] = [r(raw.latitude.min(), 4), r(raw.latitude.max(), 4)]
RESULTS["lon_range"] = [r(raw.longitude.min(), 4), r(raw.longitude.max(), 4)]
RESULTS["last_review_range"] = [raw.last_review.min(), raw.last_review.max()]
RESULTS["n_hosts"] = int(raw.host_id.nunique())
RESULTS["n_neighbourhoods"] = int(raw.neighbourhood.nunique())
RESULTS["n_names"] = int(raw.name.nunique())

# calculated_host_listings_count vs what we can count in the file
per_host = raw.groupby("host_id").size()
chlc_match = (raw.host_id.map(per_host) == raw.calculated_host_listings_count)
RESULTS["chlc_matches_file_count_pct"] = r(100 * chlc_match.mean(), 1)
RESULTS["top_host_listings"] = int(raw.calculated_host_listings_count.max())

# leakage check: rank correlation of every numeric column with the target
num_all = [c for c in raw.select_dtypes("number").columns if c not in ("id", "host_id")]
spear_target = raw[num_all].corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
RESULTS["spearman_with_price_raw"] = {k: r(v) for k, v in spear_target.items()}

# ---- target ----------------------------------------------------------------
df = load_data()
RESULTS["rows_after_filter"] = len(df)
p = df[TARGET]
RESULTS["price_stats"] = {
    "mean": r(p.mean(), 2), "median": float(p.median()), "std": r(p.std(), 2),
    "min": int(p.min()), "q1": float(p.quantile(.25)), "q3": float(p.quantile(.75)),
    "p90": float(p.quantile(.9)), "p95": float(p.quantile(.95)),
    "p99": float(p.quantile(.99)), "max": int(p.max()),
    "skew": r(p.skew()), "log_skew": r(np.log1p(p).skew()),
    "kurtosis": r(p.kurtosis(), 1),
    "share_le_1000": r(100 * (p <= 1000).mean(), 2),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(p[p <= 1000], bins=100, color=CAT[0], edgecolor="white", linewidth=0.3)
axes[0].axvline(p.median(), color=CAT[1], ls="--", lw=1.5, label=f"median = ${p.median():.0f}")
axes[0].axvline(p.mean(), color=CAT[5], ls=":", lw=1.5, label=f"mean = ${p.mean():.0f}")
axes[0].set(title=f"(a) price, listings up to $1,000 ({RESULTS['price_stats']['share_le_1000']:.1f}% of rows)",
            xlabel="price (USD / night)", ylabel="listings")
axes[0].legend()
axes[1].hist(np.log1p(p), bins=80, color=CAT[0], edgecolor="white", linewidth=0.3)
axes[1].set(title=f"(b) log1p(price), skewness = {np.log1p(p).skew():.2f}", xlabel="log1p(price)", ylabel="listings")
fig.suptitle("Figure 1 - Target distribution: raw price vs log price", y=1.02, fontsize=12)
save(fig, 1, "target_price")

# ---- split -----------------------------------------------------------------
X_train, X_test, y_train, y_test = split_data(df)
RESULTS["split"] = {"train": len(X_train), "test": len(X_test),
                    "test_frac": r(len(X_test) / len(df), 3)}
RESULTS["split_check"] = {
    "price_median": [float(y_train.median()), float(y_test.median())],
    "price_mean": [r(y_train.mean(), 1), r(y_test.mean(), 1)],
    "group_share_train": (X_train.neighbourhood_group.value_counts(normalize=True) * 100).round(1).to_dict(),
    "group_share_test": (X_test.neighbourhood_group.value_counts(normalize=True) * 100).round(1).to_dict(),
    "room_share_train": (X_train.room_type.value_counts(normalize=True) * 100).round(1).to_dict(),
    "room_share_test": (X_test.room_type.value_counts(normalize=True) * 100).round(1).to_dict(),
}
RESULTS["hoods_only_in_test"] = sorted(set(X_test.neighbourhood) - set(X_train.neighbourhood))

fig, ax = plt.subplots(figsize=(7, 3.8))
bins = np.linspace(np.log1p(df.price).min(), np.log1p(df.price).max(), 60)
ax.hist(np.log1p(y_train), bins=bins, density=True, alpha=0.6, color=CAT[0], label=f"train (n={len(y_train):,})")
ax.hist(np.log1p(y_test), bins=bins, density=True, histtype="step", lw=2, color=CAT[1], label=f"test (n={len(y_test):,})")
ax.set(title="Figure 2 - log1p(price) in train and test after the stratified split", xlabel="log1p(price)", ylabel="density")
ax.legend()
save(fig, 2, "split_check")

# everything below is TRAIN ONLY
train = X_train.copy()
train[TARGET] = y_train.values
train["log_price"] = np.log1p(y_train.values)

# ============================================================================
# 2. Univariate
# ============================================================================
desc = train[NUMERIC_FEATURES + [TARGET]].describe(percentiles=[.25, .5, .75]).T
desc["skew"] = train[NUMERIC_FEATURES + [TARGET]].skew()
desc["zeros_pct"] = (train[NUMERIC_FEATURES + [TARGET]] == 0).mean() * 100
RESULTS["numeric_describe"] = desc.round(3).to_dict(orient="index")

fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
axes = axes.ravel()
log_cols = {"minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"}
for ax, c in zip(axes, NUMERIC_FEATURES):
    s = train[c].dropna()
    if c in log_cols:
        ax.hist(np.log1p(s), bins=50, color=CAT[0], edgecolor="white", linewidth=0.3)
        ax.set_xlabel(f"log1p({c})")
    else:
        ax.hist(s, bins=50, color=CAT[0], edgecolor="white", linewidth=0.3)
        ax.set_xlabel(c)
    ax.set_title(f"skew = {s.skew():.1f}", fontsize=10)
    ax.set_ylabel("listings")
axes[-1].axis("off")
fig.suptitle("Figure 3 - Distribution of the 7 numerical features (training set; heavy-tailed ones on log1p scale)", y=1.01, fontsize=12)
save(fig, 3, "numeric_histograms")

mn = train.minimum_nights.value_counts().sort_index()
RESULTS["min_nights_top"] = {int(k): int(v) for k, v in train.minimum_nights.value_counts().head(6).items()}
RESULTS["min_nights_share_1_3"] = r(100 * train.minimum_nights.between(1, 3).mean(), 1)
RESULTS["min_nights_eq_30_pct"] = r(100 * (train.minimum_nights == 30).mean(), 1)
RESULTS["avail_zero_train_pct"] = r(100 * (train.availability_365 == 0).mean(), 1)
RESULTS["avail_365_train_pct"] = r(100 * (train.availability_365 == 365).mean(), 1)
RESULTS["reviews_zero_train_pct"] = r(100 * (train.number_of_reviews == 0).mean(), 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].bar(mn.index[mn.index <= 40], mn.values[mn.index <= 40], color=CAT[0])
axes[0].set(title="(a) minimum_nights, 1 to 40", xlabel="minimum nights", ylabel="listings")
axes[0].annotate("30-day spike", xy=(30, mn.get(30, 0)), xytext=(33, mn.get(30, 0) * 1.6),
                 arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9)
axes[1].hist(train.availability_365, bins=73, color=CAT[0], edgecolor="white", linewidth=0.3)
axes[1].set(title="(b) availability_365", xlabel="days available in the next 365", ylabel="listings")
fig.suptitle("Figure 4 - Two features with spikes that matter: minimum_nights and availability_365", y=1.03, fontsize=12)
save(fig, 4, "spikes")

# ---- categorical -------------------------------------------------------------
cat_summary = {}
for c in CATEGORICAL_FEATURES:
    vc = train[c].value_counts()
    cat_summary[c] = {
        "cardinality": int(vc.size),
        "top": vc.index[0], "top_pct": r(100 * vc.iloc[0] / len(train), 1),
        "least": vc.index[-1], "least_n": int(vc.iloc[-1]),
        "n_below_1pct": int((vc / len(train) < 0.01).sum()),
        f"n_below_{MIN_NEIGHBOURHOOD_FREQ}": int((vc < MIN_NEIGHBOURHOOD_FREQ).sum()),
    }
RESULTS["categorical"] = cat_summary
RESULTS["group_counts_train"] = train.neighbourhood_group.value_counts().to_dict()
RESULTS["group_share_train"] = (100 * train.neighbourhood_group.value_counts(normalize=True)).round(1).to_dict()
RESULTS["room_counts_train"] = train.room_type.value_counts().to_dict()
RESULTS["room_share_train"] = (100 * train.room_type.value_counts(normalize=True)).round(1).to_dict()
vc_h = train.neighbourhood.value_counts()
RESULTS["hood_top10_share"] = r(100 * vc_h.head(10).sum() / len(train), 1)
RESULTS["hood_top25_share"] = r(100 * vc_h.head(25).sum() / len(train), 1)
RESULTS["hood_n_ge_30"] = int((vc_h >= MIN_NEIGHBOURHOOD_FREQ).sum())
RESULTS["hood_rows_in_rare"] = int(vc_h[vc_h < MIN_NEIGHBOURHOOD_FREQ].sum())
RESULTS["hood_rows_in_rare_pct"] = r(100 * vc_h[vc_h < MIN_NEIGHBOURHOOD_FREQ].sum() / len(train), 2)
RESULTS["hood_single_listing"] = int((vc_h == 1).sum())
RESULTS["hood_top5"] = vc_h.head(5).to_dict()

ct = pd.crosstab(train.neighbourhood_group, train.room_type).loc[GROUP_ORDER, ROOM_ORDER]
RESULTS["crosstab_group_room"] = ct.to_dict()
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(GROUP_ORDER)); w = 0.27
for i, rt in enumerate(ROOM_ORDER):
    ax.bar(x + (i - 1) * w, ct[rt], width=w, color=ROOM_COL[rt], label=rt)
    for xi, v in zip(x + (i - 1) * w, ct[rt]):
        ax.text(xi, v + 150, f"{v:,}", ha="center", fontsize=7.5, color="0.3")
ax.set_xticks(x, GROUP_ORDER)
ax.set(title="Figure 5 - Listings per borough and room type (training set)", ylabel="listings", xlabel="neighbourhood_group")
ax.legend(title="room_type")
save(fig, 5, "borough_roomtype")

top = vc_h.head(25)
hood_group = train.drop_duplicates("neighbourhood").set_index("neighbourhood").neighbourhood_group
fig, ax = plt.subplots(figsize=(8, 7))
colors = [GROUP_COL[hood_group[h]] for h in top.index]
ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
for i, v in enumerate(top.values[::-1]):
    ax.text(v + 30, i, f"{v:,}", va="center", fontsize=8, color="0.3")
handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COL[g]) for g in GROUP_ORDER]
ax.legend(handles, GROUP_ORDER, title="borough", loc="lower right")
ax.set(title=f"Figure 6 - The 25 most frequent neighbourhoods (of {vc_h.size}) hold {RESULTS['hood_top25_share']}% of the training listings",
       xlabel="listings")
save(fig, 6, "top_neighbourhoods")

# ============================================================================
# 3. Bivariate
# ============================================================================
cols = NUMERIC_FEATURES + [TARGET]
sp = train[cols].corr(method="spearman")
pe = train[cols].corr(method="pearson")
pairs = []
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        pairs.append((a, b, sp.loc[a, b], pe.loc[a, b]))
pairs_df = pd.DataFrame(pairs, columns=["a", "b", "spearman", "pearson"])
pairs_df["abs_s"] = pairs_df.spearman.abs()
pairs_df = pairs_df.sort_values("abs_s", ascending=False)
RESULTS["top_pairs"] = pairs_df.head(8).drop(columns="abs_s").round(3).to_dict(orient="records")
feat_pairs = pairs_df[(pairs_df.a != TARGET) & (pairs_df.b != TARGET)]
RESULTS["most_correlated_feature_pair"] = feat_pairs.drop(columns="abs_s").round(3).iloc[0].to_dict()
RESULTS["spearman_with_price_train"] = sp[TARGET].drop(TARGET).round(3).to_dict()
RESULTS["pearson_with_price_train"] = pe[TARGET].drop(TARGET).round(3).to_dict()

fig, ax = plt.subplots(figsize=(8, 6.5))
mask = np.triu(np.ones_like(sp, dtype=bool), k=1)
sns.heatmap(sp, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Spearman rho"}, ax=ax)
ax.set_title("Figure 7 - Spearman correlation, numerical features + price (training set)")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
save(fig, 7, "spearman_heatmap")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
s = train.sample(8000, random_state=RANDOM_STATE)
axes[0].scatter(s.number_of_reviews, s.reviews_per_month.fillna(0), s=6, alpha=0.3, color=CAT[0])
axes[0].set(xscale="symlog", yscale="symlog", xlabel="number_of_reviews", ylabel="reviews_per_month",
            title=f"(a) rho = {sp.loc['number_of_reviews', 'reviews_per_month']:.2f}: the same information twice")
axes[1].scatter(s.availability_365, s.calculated_host_listings_count, s=6, alpha=0.3, color=CAT[0])
axes[1].set(yscale="symlog", xlabel="availability_365", ylabel="calculated_host_listings_count",
            title=f"(b) rho = {sp.loc['availability_365', 'calculated_host_listings_count']:.2f}: multi-listing hosts keep calendars open")
fig.suptitle("Figure 8 - The two strongest feature-feature relations (8,000 training rows)", y=1.03, fontsize=12)
save(fig, 8, "scatter_pairs")

fig, ax = plt.subplots(figsize=(7.5, 7))
s = train.sample(15000, random_state=RANDOM_STATE).sort_values("log_price")
sc = ax.scatter(s.longitude, s.latitude, c=s.log_price, cmap=SEQ, s=4, alpha=0.7, vmin=VMIN, vmax=VMAX)
price_colorbar(sc, ax)
ax.set(title="Figure 9 - Where the listings are and what they cost (15,000 training rows)", xlabel="longitude", ylabel="latitude")
ax.set_aspect(1.3)
save(fig, 9, "map_price")

# ---- categorical x target ----------------------------------------------------
med = train.groupby(["neighbourhood_group", "room_type"])[TARGET].median().unstack().loc[GROUP_ORDER, ROOM_ORDER]
RESULTS["median_price_group_room"] = med.to_dict()
RESULTS["median_price_group"] = train.groupby("neighbourhood_group")[TARGET].median().loc[GROUP_ORDER].to_dict()
RESULTS["median_price_room"] = train.groupby("room_type")[TARGET].median().loc[ROOM_ORDER].to_dict()
RESULTS["iqr_log_price_room"] = (train.groupby("room_type").log_price.quantile(.75) - train.groupby("room_type").log_price.quantile(.25)).round(2).loc[ROOM_ORDER].to_dict()
RESULTS["iqr_log_price_group"] = (train.groupby("neighbourhood_group").log_price.quantile(.75) - train.groupby("neighbourhood_group").log_price.quantile(.25)).round(2).loc[GROUP_ORDER].to_dict()

fig, ax = plt.subplots(figsize=(11, 4.6))
sns.boxplot(data=train, x="neighbourhood_group", y="log_price", hue="room_type", order=GROUP_ORDER,
            hue_order=ROOM_ORDER, palette=ROOM_COL, fliersize=1.5, linewidth=0.9, ax=ax)
yt = [25, 50, 100, 200, 400, 800, 1600]
ax.set_yticks(np.log1p(yt)); ax.set_yticklabels([f"${v}" for v in yt])
ax.set(title="Figure 10 - Price per borough and room type (training set, log scale)", ylabel="price (log scale)", xlabel="neighbourhood_group")
ax.legend(title="room_type", loc="upper right")
save(fig, 10, "price_by_borough_room")

hood_stats = train.groupby("neighbourhood")[TARGET].agg(["median", "size"])
hood_stats = hood_stats[hood_stats["size"] >= MIN_NEIGHBOURHOOD_FREQ].sort_values("median")
RESULTS["hood_median_top5"] = hood_stats.tail(5)["median"].to_dict()
RESULTS["hood_median_bottom5"] = hood_stats.head(5)["median"].to_dict()
RESULTS["hood_median_ratio"] = r(hood_stats["median"].max() / hood_stats["median"].min(), 1)
sel = pd.concat([hood_stats.head(12), hood_stats.tail(12)])
fig, ax = plt.subplots(figsize=(8, 7))
colors = [GROUP_COL[hood_group[h]] for h in sel.index]
ax.barh(sel.index, sel["median"], color=colors)
for i, (v, n) in enumerate(zip(sel["median"], sel["size"])):
    ax.text(v + 4, i, f"${v:.0f}  (n={n})", va="center", fontsize=7.5, color="0.3")
handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COL[g]) for g in GROUP_ORDER]
ax.legend(handles, GROUP_ORDER, title="borough", loc="lower right")
ax.set(title=f"Figure 11 - Median price: 12 cheapest and 12 most expensive neighbourhoods (>= {MIN_NEIGHBOURHOOD_FREQ} listings)", xlabel="median price (USD)")
ax.set_xlim(0, sel["median"].max() * 1.3)
save(fig, 11, "hood_median_price")

# ---- numerical x categorical ------------------------------------------------
def grp_stats(col, by):
    g = train.groupby(by)[col]
    return pd.DataFrame({"median": g.median(), "q1": g.quantile(.25), "q3": g.quantile(.75)}).round(2)

RESULTS["avail_by_room"] = grp_stats("availability_365", "room_type").loc[ROOM_ORDER].to_dict(orient="index")
RESULTS["minn_by_room"] = grp_stats("minimum_nights", "room_type").loc[ROOM_ORDER].to_dict(orient="index")
RESULTS["reviews_by_group"] = grp_stats("number_of_reviews", "neighbourhood_group").loc[GROUP_ORDER].to_dict(orient="index")
RESULTS["chlc_by_room"] = grp_stats("calculated_host_listings_count", "room_type").loc[ROOM_ORDER].to_dict(orient="index")
RESULTS["chlc_gt1_by_room"] = (train.groupby("room_type").calculated_host_listings_count.apply(lambda s: 100 * (s > 1).mean())).round(1).loc[ROOM_ORDER].to_dict()
RESULTS["avail_zero_by_room"] = (train.groupby("room_type").availability_365.apply(lambda s: 100 * (s == 0).mean())).round(1).loc[ROOM_ORDER].to_dict()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
sns.boxplot(data=train, x="room_type", y="availability_365", order=ROOM_ORDER, palette=ROOM_COL, hue="room_type", legend=False, fliersize=1.5, linewidth=0.9, ax=axes[0])
axes[0].set(title="(a) availability_365 by room_type", xlabel="room_type", ylabel="days available")
tmp = train.assign(log_mn=np.log1p(train.minimum_nights))
sns.boxplot(data=tmp, x="room_type", y="log_mn", order=ROOM_ORDER, palette=ROOM_COL, hue="room_type", legend=False, fliersize=1.5, linewidth=0.9, ax=axes[1])
yt = [1, 2, 3, 5, 10, 30, 100, 365]
axes[1].set_yticks(np.log1p(yt)); axes[1].set_yticklabels(yt)
axes[1].set(title="(b) minimum_nights by room_type (log scale)", xlabel="room_type", ylabel="minimum nights")
fig.suptitle("Figure 12 - Availability and minimum stay per room type (training set)", y=1.03, fontsize=12)
save(fig, 12, "box_room_type")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
tmp = train.assign(log_rev=np.log1p(train.number_of_reviews), log_chlc=np.log1p(train.calculated_host_listings_count))
sns.boxplot(data=tmp, x="neighbourhood_group", y="log_rev", order=GROUP_ORDER, palette=GROUP_COL, hue="neighbourhood_group", legend=False, fliersize=1.5, linewidth=0.9, ax=axes[0])
yt = [0, 1, 3, 10, 30, 100, 300]
axes[0].set_yticks(np.log1p(yt)); axes[0].set_yticklabels(yt)
axes[0].set(title="(a) number_of_reviews by borough (log scale)", xlabel="neighbourhood_group", ylabel="reviews")
plt.setp(axes[0].get_xticklabels(), rotation=15)
sns.boxplot(data=tmp, x="room_type", y="log_chlc", order=ROOM_ORDER, palette=ROOM_COL, hue="room_type", legend=False, fliersize=1.5, linewidth=0.9, ax=axes[1])
yt = [1, 2, 3, 5, 10, 30, 100, 300]
axes[1].set_yticks(np.log1p(yt)); axes[1].set_yticklabels(yt)
axes[1].set(title="(b) calculated_host_listings_count by room_type (log scale)", xlabel="room_type", ylabel="listings of the same host")
fig.suptitle("Figure 13 - Reviews per borough and host portfolio size per room type (training set)", y=1.03, fontsize=12)
save(fig, 13, "box_group")

# ============================================================================
# 4. Preprocessing
# ============================================================================
pre = build_preprocessor()
X_train_t = pre.fit_transform(X_train)
X_test_t = pre.transform(X_test)
names = list(pre.get_feature_names_out())
RESULTS["pipeline"] = {
    "train_shape": list(X_train_t.shape), "test_shape": list(X_test_t.shape),
    "nan_train": int(np.isnan(X_train_t).sum()), "nan_test": int(np.isnan(X_test_t).sum()),
    "n_features": len(names),
    "n_hood_columns": sum(n.startswith("hood__") for n in names),
    "feature_names_head": names[:15],
}
clipper = pre.named_transformers_["heavy"].named_steps["clip"]
imputed_train = pre.named_transformers_["heavy"].named_steps["impute"].transform(X_train[HEAVY_TAIL_FEATURES])
RESULTS["clip_bounds"] = {c: [float(lo), float(hi)] for c, lo, hi in zip(HEAVY_TAIL_FEATURES, clipper.low_, clipper.high_)}
RESULTS["clip_rows_affected_train"] = clipper.rows_affected(imputed_train)
RESULTS["clip_rows_affected_train_pct"] = r(100 * clipper.rows_affected(imputed_train) / len(X_train), 2)
RESULTS["clip_rows_affected_per_col"] = {
    c: int(((imputed_train[:, i] < clipper.low_[i]) | (imputed_train[:, i] > clipper.high_[i])).sum())
    for i, c in enumerate(HEAVY_TAIL_FEATURES)
}
imputed_test = pre.named_transformers_["heavy"].named_steps["impute"].transform(X_test[HEAVY_TAIL_FEATURES])
RESULTS["clip_rows_affected_test"] = clipper.rows_affected(imputed_test)
ohe_h = pre.named_transformers_["hood"].named_steps["onehot"]
RESULTS["hood_infrequent_n"] = int(len(ohe_h.infrequent_categories_[0]))
RESULTS["test_rows_in_infrequent"] = int(X_test_t[:, names.index("hood__neighbourhood_infrequent_sklearn")].sum())

# before / after for the heavy-tailed block
after = pd.DataFrame(X_train_t[:, [names.index(f"heavy__{c}") for c in HEAVY_TAIL_FEATURES]], columns=HEAVY_TAIL_FEATURES)
RESULTS["heavy_after_skew"] = after.skew().round(2).to_dict()
RESULTS["heavy_before_skew"] = train[HEAVY_TAIL_FEATURES].fillna(0).skew().round(2).to_dict()
fig, axes = plt.subplots(2, 4, figsize=(14, 5.6))
for j, c in enumerate(HEAVY_TAIL_FEATURES):
    axes[0, j].hist(train[c].fillna(0), bins=60, color=CAT[0], edgecolor="white", linewidth=0.3)
    axes[0, j].set(title=f"{c}\nraw, skew {train[c].fillna(0).skew():.1f}", ylabel="listings (log)" if j == 0 else "")
    axes[0, j].set_yscale("log")
    axes[1, j].hist(after[c], bins=60, color=CAT[2], edgecolor="white", linewidth=0.3)
    axes[1, j].set(title=f"clip 1-99% + log1p + z-score, skew {after[c].skew():.1f}", ylabel="listings" if j == 0 else "", xlabel="standardized value")
fig.suptitle("Figure 14 - Heavy-tailed features before (top, log count axis) and after the numeric pipeline (bottom)", y=1.02, fontsize=12)
save(fig, 14, "before_after")

# ---- PCA (pipeline output minus the sparse neighbourhood dummies) --------------
keep = [i for i, n in enumerate(names) if not n.startswith("hood__")]
Z = X_train_t[:, keep]
Z_names = [names[i].split("__", 1)[1] for i in keep]
pca = PCA(random_state=RANDOM_STATE).fit(Z)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
RESULTS["pca"] = {
    "n_input_features": Z.shape[1],
    "evr": [r(v, 4) for v in evr],
    "pc1_pc2": r(100 * cum[1], 1),
    "n_for_80": int(np.searchsorted(cum, 0.80) + 1),
    "n_for_95": int(np.searchsorted(cum, 0.95) + 1),
}
load = pd.DataFrame(pca.components_[:3].T, index=Z_names, columns=["PC1", "PC2", "PC3"])
RESULTS["pca_loadings"] = load.round(3).to_dict()
Zp = pca.transform(Z)
# PCA on the 7 scaled numerical features alone, as a reference
pca_num = PCA(random_state=RANDOM_STATE).fit(Z[:, :7])
RESULTS["pca_numeric_only_pc1_pc2"] = r(100 * np.cumsum(pca_num.explained_variance_ratio_)[1], 1)
RESULTS["pca_pc1_spearman_price"] = r(pd.Series(Zp[:, 0]).corr(pd.Series(train.log_price.values), method="spearman"))
RESULTS["pca_pc2_spearman_price"] = r(pd.Series(Zp[:, 1]).corr(pd.Series(train.log_price.values), method="spearman"))

rng = np.random.RandomState(RANDOM_STATE)
idx = rng.choice(len(Z), 5000, replace=False)
lp = train.log_price.values

fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.35]})
axes[0].bar(np.arange(1, len(evr) + 1), 100 * evr, color=CAT[0], label="per component")
axes[0].plot(np.arange(1, len(evr) + 1), 100 * cum, "o-", color=CAT[1], ms=4, label="cumulative")
axes[0].axhline(80, color="0.5", ls=":", lw=1); axes[0].axhline(95, color="0.5", ls=":", lw=1)
axes[0].set(title=f"(a) explained variance; PC1+PC2 = {100*cum[1]:.1f}%", xlabel="component", ylabel="% of variance")
axes[0].legend()
order = np.argsort(lp[idx])
sc = axes[1].scatter(Zp[idx][order, 0], Zp[idx][order, 1], c=lp[idx][order], cmap=SEQ, s=6, alpha=0.7, vmin=VMIN, vmax=VMAX)
price_colorbar(sc, axes[1])
axes[1].set(title="(b) PC1 x PC2, 5,000 training rows coloured by price", xlabel=f"PC1 ({100*evr[0]:.1f}%)", ylabel=f"PC2 ({100*evr[1]:.1f}%)")
fig.suptitle(f"Figure 15 - PCA on the {Z.shape[1]} scaled pipeline features (numerical + borough + room type dummies)", y=1.02, fontsize=12)
save(fig, 15, "pca")

fig, ax = plt.subplots(figsize=(8, 5.2))
sns.heatmap(load, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-0.7, vmax=0.7, center=0, linewidths=0.5, cbar_kws={"label": "loading"}, ax=ax)
ax.set_title("Figure 16 - PCA loadings of the first three components")
save(fig, 16, "pca_loadings")

# ---- t-SNE -------------------------------------------------------------------
Zs = Z[idx]; lps = lp[idx]
order = np.argsort(lps)
perps = [30, 100]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
tsne_time = {}
for ax, pp in zip(axes, perps):
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=pp, init="pca", learning_rate="auto", random_state=RANDOM_STATE).fit_transform(Zs)
    tsne_time[pp] = round(time.time() - t0, 1)
    sc = ax.scatter(emb[order, 0], emb[order, 1], c=lps[order], cmap=SEQ, s=6, alpha=0.75, vmin=VMIN, vmax=VMAX)
    ax.set(title=f"perplexity = {pp}", xlabel="t-SNE 1", ylabel="t-SNE 2", xticks=[], yticks=[])
price_colorbar(sc, axes)
fig.suptitle("Figure 17 - t-SNE of 5,000 training rows (same features as the PCA), coloured by price", y=0.98, fontsize=12)
save(fig, 17, "tsne", tight=False)
RESULTS["tsne_seconds"] = tsne_time

# ---- UMAP --------------------------------------------------------------------
import umap  # noqa: E402  (slow import, keep it here)
nns = [15, 100]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
umap_time = {}
for ax, nn in zip(axes, nns):
    t0 = time.time()
    emb = umap.UMAP(n_neighbors=nn, min_dist=0.1, n_components=2, random_state=RANDOM_STATE).fit_transform(Zs)
    umap_time[nn] = round(time.time() - t0, 1)
    sc = ax.scatter(emb[order, 0], emb[order, 1], c=lps[order], cmap=SEQ, s=6, alpha=0.75, vmin=VMIN, vmax=VMAX)
    ax.set(title=f"n_neighbors = {nn}, min_dist = 0.1", xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
price_colorbar(sc, axes)
fig.suptitle("Figure 18 - UMAP of the same 5,000 training rows, coloured by price", y=0.98, fontsize=12)
save(fig, 18, "umap", tight=False)
RESULTS["umap_seconds"] = umap_time

# how much of the log-price variance do room_type + borough explain alone? (for the synthesis)
grp_mean = train.groupby(["neighbourhood_group", "room_type"]).log_price.transform("mean")
RESULTS["r2_log_price_room_borough_means"] = r(1 - ((train.log_price - grp_mean) ** 2).sum() / ((train.log_price - train.log_price.mean()) ** 2).sum(), 3)
grp_mean2 = train.groupby(["neighbourhood", "room_type"]).log_price.transform("mean")
RESULTS["r2_log_price_room_hood_means"] = r(1 - ((train.log_price - grp_mean2) ** 2).sum() / ((train.log_price - train.log_price.mean()) ** 2).sum(), 3)

out = ROOT / "docs" / "projects" / "eda" / "results.json"
out.write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
print(f"wrote {out}")
