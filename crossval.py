import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import shap
from config import SEASON_FEATURES, TEST_YEARS, PARQUET_PATH
from model import make_model
from features import build_features, build_season_table, load_surgeries
 
 
def cross_validate(season, features=None):
    features = features or SEASON_FEATURES
    pr_scores, roc_scores = [], []
    shap_dir = {f: [] for f in features} # track each feature's direction per fold

    for year in TEST_YEARS:
        train = season[season["season"] < year]
        test = season[season["season"] == year]
        if test["TJ"].sum() == 0 or train["TJ"].sum() == 0:
            print(f"{year}: skipped (no injuries in train or test)")
            continue

        X_tr, y_tr = train[features], train["TJ"]
        X_te, y_te = test[features], test["TJ"]

        model = make_model(y_tr)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        pr = average_precision_score(y_te, proba)
        roc = roc_auc_score(y_te, proba)
        pr_scores.append(pr); roc_scores.append(roc)

        sv = shap.TreeExplainer(model).shap_values(X_te)
        sv = sv[1] if isinstance(sv, list) else sv
        for i, f in enumerate(features):
            vals = X_te[f].values.astype(float)
            s = sv[:, i]
            m = ~np.isnan(vals)
            if m.sum() > 2 and np.std(vals[m]) > 0 and np.std(s[m]) > 0:
                shap_dir[f].append(np.sign(np.corrcoef(vals[m], s[m])[0, 1]))

        print(f"{year}: PR-AUC {pr:.3f} (random {y_te.mean():.3f})  ROC-AUC {roc:.3f}  "f"[{int(y_te.sum())} injuries / {len(y_te)} seasons]")

    print("\n" + "=" * 50)
    print(f"PR-AUC : {np.mean(pr_scores):.3f} +/- {np.std(pr_scores):.3f}  "f"folds={[round(s, 3) for s in pr_scores]}")
    print(f"ROC-AUC: {np.mean(roc_scores):.3f} +/- {np.std(roc_scores):.3f}")

    print("\nfeature direction stability (mean sign across folds):")
    rows = [(f, np.mean(s), len(s)) for f, s in shap_dir.items() if s]
    for f, sign, n in sorted(rows, key=lambda r: -abs(r[1])):
        verdict = ("raises risk" if sign > 0.6 else "lowers risk" if sign < -0.6 else "UNSTABLE")
        print(f"  {f:16s} {sign:+.2f}  ({verdict}, {n} folds)")

if __name__ == "__main__":
    season = build_season_table(build_features(pd.read_parquet(PARQUET_PATH)), load_surgeries())
    cross_validate(season)