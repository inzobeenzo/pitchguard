 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve, roc_curve, ConfusionMatrixDisplay)
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

df = pd.read_parquet("statcast_2015_2023.parquet")
print(df.head())
FASTBALLS = {"FF", "SI", "FT", "FC"}              # four-seam, sinker, two-seam, cutter
BREAKING  = {"SL", "CU", "KC", "ST", "SV", "CS"}  # slider, curve, knuckle-curve, sweeper, etc.
 
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    is_fb = df["pitch_type"].isin(FASTBALLS)
    df["fb_speed"] = df["release_speed"].where(is_fb)
    df["fb_eff_speed"] = df["effective_speed"].where(is_fb)
    df["is_breaking"] = df["pitch_type"].isin(BREAKING).astype(float)
    df["is_fastball"] = df["pitch_type"].isin(FASTBALLS).astype(float)

    outings = (
        df.groupby(["pitcher", "player_name", "game_date"])
          .agg(
            pitch_count = ("release_speed", "size"),
            fb_velo_mean = ("fb_speed", "mean"),
            fb_velo_max = ("fb_speed", "max"),
            fb_eff_velo_mean = ("fb_eff_speed", "mean"),
            spin_mean = ("release_spin_rate", "mean"),
            pfx_x_mean = ("pfx_x", "mean"),
            pfx_z_mean = ("pfx_z", "mean"),
            rel_x_mean = ("release_pos_x", "mean"),
            rel_z_mean = ("release_pos_z", "mean"),
            rel_x_std = ("release_pos_x", "std"),
            rel_z_std = ("release_pos_z", "std"),
            arm_angle_mean = ("arm_angle", "mean"),
            breaking_pct = ("is_breaking", "mean"),
            fastball_pct = ("is_fastball", "mean"),
            arsenal_size = ("pitch_type", "nunique"),
            days_rest = ("pitcher_days_since_prev_game", "first"),
            age = ("age_pit", "first"),
          )
          .reset_index()
          .sort_values(["pitcher", "game_date"])
    )

    def per_pitcher(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("game_date").sort_index()
 
        # ACWR: acute (last 7 days) load vs chronic (last 28 days) baseline.
        # Divide chronic by 4 so both are on a per-WEEK scale; otherwise
        # the ratio is always tiny. >1.5 is "danger" zone.
        acute = g["pitch_count"].rolling("7D").sum()
        chronic = g["pitch_count"].rolling("28D").sum() / 4
        g["acwr"] = acute / chronic
 
        # Velocity TREND: recent 3-start average minus
        # the pitcher's running baseline. Negative => velocity dropping
        # (fatigue signal); sustained positive => overexertion.
        recent_velo = g["fb_velo_mean"].rolling(3, min_periods=1).mean()
        baseline_velo = g["fb_velo_mean"].shift(1).expanding(min_periods=1).mean()
        g["velo_trend"] = recent_velo - baseline_velo
 
        # Release-point DRIFT: distance of this outing's release point from
        # the pitcher's own baseline (drift is a flagged early warning sign).
        base_x = g["rel_x_mean"].shift(1).expanding(min_periods=1).mean()
        base_z = g["rel_z_mean"].shift(1).expanding(min_periods=1).mean()
        g["release_drift"] = np.sqrt((g["rel_x_mean"] - base_x) ** 2 + (g["rel_z_mean"] - base_z) ** 2)
 
        return g.reset_index()
 
    features = outings.groupby("pitcher", group_keys=True).apply(per_pitcher)
    return features.reset_index(level="pitcher", drop=False)
 
features = build_features(df)

def add_injury_labels(features: pd.DataFrame, surgeries: pd.DataFrame, N=365) -> pd.DataFrame:
    features["game_date"] = pd.to_datetime(features["game_date"])
    merged = pd.merge(
        features[["pitcher", "game_date"]],
        surgeries,
        left_on = "pitcher",
        right_on = "pitcher_id",
        how = "left"
    )
    days_until_surgery = (merged["surgery_date"] - merged["game_date"]).dt.days
    merged["is_injured_outing"] = (days_until_surgery <= N) & (days_until_surgery >= 0)
    injury_labels = (
        merged.groupby(["pitcher", "game_date"])["is_injured_outing"].any().astype(int).reset_index(name="TJ")
    )

    features = pd.merge(features, injury_labels, on=["pitcher", "game_date"], how="left")
    features["TJ"] = features["TJ"].fillna(0).astype(int)
    return features

sheet_url = "https://docs.google.com/spreadsheets/u/1/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/edit"
surgery_raw = pd.read_csv(sheet_url.replace("/edit", "/export?format=csv"), skiprows=1)
surgeries = surgery_raw[["mlbamid", "TJ Surgery Date"]].dropna()
surgeries.columns = ["pitcher_id", "surgery_date"]
surgeries["surgery_date"] = pd.to_datetime(surgeries["surgery_date"])

features = add_injury_labels(features, surgeries, N=365)

surg = surgeries.rename(columns={"pitcher_id": "pitcher"})
surg_dates = surg.groupby("pitcher")["surgery_date"].apply(list).to_dict()
 

# A pitcher who had 30 outings in 2019 becomes ONE row summarizing 2019.
# For each summary we deliberately pick the statistic that carries signal:
# sum  -> total workload      max -> single worst spike
# min  -> worst velocity drop  mean -> his typical level
features = features.copy()
features["game_date"] = pd.to_datetime(features["game_date"])
features["season"] = features["game_date"].dt.year
 
season = (
    features.groupby(["pitcher", "season"])
    .agg(
        outings = ("game_date",      "size"),   # how many appearances
        total_pitches = ("pitch_count",    "sum"),    # full-season workload
        max_pitches = ("pitch_count",    "max"),    # hardest single outing
        avg_pitches = ("pitch_count",    "mean"),   # ROLE: high=starter, low=reliever
        fb_velo_mean = ("fb_velo_mean",   "mean"),
        velo_decline = ("velo_trend",     "min"),    # most negative = worst drop
        acwr_peak = ("acwr",           "max"),    # biggest workload spike
        drift_max = ("release_drift",  "max"),    # biggest mechanical drift
        spin_mean = ("spin_mean",      "mean"),
        breaking_pct = ("breaking_pct",   "mean"),
        fastball_pct = ("fastball_pct",   "mean"),
        arsenal_size = ("arsenal_size",   "max"),
        rel_x_std = ("rel_x_std",      "mean"),
        rel_z_std = ("rel_z_std",      "mean"),
        arm_angle_mean = ("arm_angle_mean", "mean"),
        days_rest_mean = ("days_rest",      "mean"),
        age = ("age",            "max"),
    )
    .reset_index()
)

# STEP 2 — label each pitcher-season + add surgery history.
#   TJ = 1 if a surgery happens within 18 months AFTER the season starts
#   prior_tj = how many surgeries the pitcher already had BEFORE this season
# (Season "starts" March 1, the rough start of the MLB season.)

def season_start(year):
    return pd.Timestamp(year=int(year), month=3, day=1)
 
def make_label(row, horizon_months=18):
    start = season_start(row["season"])
    end = start + pd.DateOffset(months=horizon_months)
    dates = surg_dates.get(row["pitcher"], [])
    return int(any(start <= d <= end for d in dates)) # any surgery in the window?
 
def make_prior_tj(row):
    start = season_start(row["season"])
    dates = surg_dates.get(row["pitcher"], [])
    return sum(d < start for d in dates) # count surgeries before now
 
season["TJ"] = season.apply(make_label, axis=1)
season["prior_tj"] = season.apply(make_prior_tj, axis=1)
print(season["prior_tj"].value_counts())

# STEP 3 — the feature list, and a TIME-AWARE split into 3 groups:
#   train (old seasons) -> the model learns here
#   val (one season) -> used ONLY to pick the threshold
#   test (newest seasons)-> the final scoreboard

SEASON_FEATURES = [
    "outings", "total_pitches", "max_pitches", "avg_pitches",
    "fb_velo_mean", "velo_decline", "acwr_peak", "drift_max",
    "spin_mean", "breaking_pct", "fastball_pct", "arsenal_size",
    "rel_x_std", "rel_z_std", "arm_angle_mean", "days_rest_mean",
    "age", "prior_tj",
]
 
train = season[season["season"] <= 2020]
val = season[season["season"] == 2021]
test = season[season["season"].isin([2022, 2023])]
 
X_train, y_train = train[SEASON_FEATURES], train["TJ"]
X_val, y_val = val[SEASON_FEATURES],   val["TJ"]
X_test, y_test = test[SEASON_FEATURES],  test["TJ"]

# STEP 4 — baseline (logistic regression), then the real model (XGBoost).
# Fill missing values with the MEDIAN, not 0: for trend features that sit
# near 0, a literal 0 would be a fake reading.

med = X_train.median()
base = LogisticRegression(max_iter=1000, class_weight="balanced")
base.fit(X_train.fillna(med), y_train)
base_pr = average_precision_score(y_test, base.predict_proba(X_test.fillna(med))[:, 1])
 
pos = int(y_train.sum()); neg = len(y_train) - pos
model = xgb.XGBClassifier(
    n_estimators=150, max_depth=2, learning_rate=0.03,
    min_child_weight=5, reg_lambda=3.0, # heavier regularization
    subsample=0.7, colsample_bytree=0.7,
    scale_pos_weight=np.sqrt(neg / max(pos, 1)), # gentler than the full ratio
    eval_metric="aucpr", n_jobs=-1,
)
model.fit(X_train, y_train) # XGBoost handles NaNs itself

# STEP 5 — pick the threshold on VALIDATION, then judge on TEST.
if y_val.sum() >= 1:
    val_proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, val_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_t = float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5 # align f1 with thr
else:
    best_t = 0.5 # not enough positive cases in val to tune; fall back
 
test_proba = model.predict_proba(X_test)[:, 1]
preds = (test_proba >= best_t).astype(int)
 
pr_auc = average_precision_score(y_test, test_proba)
roc_auc = roc_auc_score(y_test, test_proba)
p, r, f, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
 
print(f"positives  train:{int(y_train.sum())}  val:{int(y_val.sum())}  test:{int(y_test.sum())}")
print(f"baseline (logreg) PR-AUC: {base_pr:.3f}")
print(f"XGBoost PR-AUC: {pr_auc:.3f} ROC-AUC: {roc_auc:.3f}")
print(f"threshold {best_t:.3f} -> precision: {p:.3f}  recall: {r:.3f}  f1: {f:.3f}")
print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, preds))

# STEP 6 — which features drive risk overall? (sanity check + RAG input)
# If prior_tj, acwr_peak, velo_decline rank high, model looks good.

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test)
sv = sv[1] if isinstance(sv, list) else sv
importance = (pd.Series(np.abs(sv).mean(axis=0), index=SEASON_FEATURES).sort_values(ascending=False))
print("\ntop risk features (mean |SHAP|):")
print(importance.head(6).to_string())

base_rate = y_test.mean() # a "random" model scores this on PR
prec, rec, _ = precision_recall_curve(y_test, test_proba)
fpr, tpr, _  = roc_curve(y_test, test_proba)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# 1) PRECISION-RECALL CURVE: the honest one. Every point is a different
#    threshold; the dashed line is random. Curve above the line = real signal.
ax[0].plot(rec, prec, lw=2, label=f"model (PR-AUC={pr_auc:.3f})")
ax[0].axhline(base_rate, ls="--", color="gray", label=f"random ({base_rate:.3f})")
ax[0].set(xlabel="Recall (injuries caught)", ylabel="Precision (flags that are right)", title="Precision-Recall curve", xlim=(0, 1), ylim=(0, 1))
ax[0].legend()

# 2) ROC CURVE: how well injured pitchers are ranked above safe ones.
#    Dashed diagonal = random. Higher/left = better.
ax[1].plot(fpr, tpr, lw=2, label=f"model (ROC-AUC={roc_auc:.3f})")
ax[1].plot([0, 1], [0, 1], ls="--", color="gray", label="random")
ax[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1))
ax[1].legend()

# 3) CONFUSION MATRIX: the four outcome boxes at your chosen threshold.
ConfusionMatrixDisplay(
    confusion_matrix(y_test, preds), display_labels=["safe", "injured"]
).plot(ax=ax[2], colorbar=False, cmap="Blues")
ax[2].set_title("Confusion matrix")

plt.tight_layout()
plt.savefig("pitchguard_metrics.png", dpi=130) 
plt.show()

# 4) SHAP SUMMARY (beeswarm): each dot is a pitcher-season. Color = feature
#    value (red high / blue low), x-position = how much it pushed risk up/down.
#    This shows not just WHICH features matter but in WHICH direction.
shap.summary_plot(sv, X_test, show=False)
plt.tight_layout()
plt.savefig("pitchguard_shap.png", dpi=130)
plt.show()