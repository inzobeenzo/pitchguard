import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve)
import xgboost as xgb
import shap
from config import SEASON_FEATURES

def make_model(y_train):
    """The XGBoost configuration, defined in ONE place."""
    pos = int(y_train.sum()); neg = len(y_train) - pos
    return xgb.XGBClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.03,
        min_child_weight=5, reg_lambda=3.0,
        subsample=0.7, colsample_bytree=0.7,
        scale_pos_weight=np.sqrt(neg / max(pos, 1)),
        eval_metric="aucpr", n_jobs=-1,
    )

def train_model(season, features=None):
    """Train on <=2020, tune the threshold on 2021, score on 2022-23.
    Returns a dict with the model + everything needed to evaluate or plot."""
    features = features or SEASON_FEATURES
    train = season[season["season"] <= 2020]
    val = season[season["season"] == 2021]
    test = season[season["season"].isin([2022, 2023])]

    X_train, y_train = train[features], train["TJ"]
    X_val, y_val = val[features],   val["TJ"]
    X_test, y_test = test[features],  test["TJ"]

    # baseline: logistic regression, median-imputed
    med = X_train.median()
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    base.fit(X_train.fillna(med), y_train)
    base_pr = average_precision_score(y_test, base.predict_proba(X_test.fillna(med))[:, 1])

    # main model
    model = make_model(y_train)
    model.fit(X_train, y_train)

    # pick the decision threshold on VALIDATION
    if y_val.sum() >= 1:
        vproba = model.predict_proba(X_val)[:, 1]
        prec, rec, thr = precision_recall_curve(y_val, vproba)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        threshold = float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5
    else:
        threshold = 0.5

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    pr_auc  = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    print(f"baseline (logreg) PR-AUC: {base_pr:.3f}")
    print(f"XGBoost PR-AUC: {pr_auc:.3f} ROC-AUC: {roc_auc:.3f}")
    print(f"threshold {threshold:.3f} -> precision {p:.3f} recall {r:.3f} f1 {f:.3f}")
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, preds))

    return {"model": model, "X_test": X_test, "y_test": y_test, "proba": proba, "preds": preds, "threshold": threshold}
 

def explain_prediction(model, X, row_idx=0, top_n=3):
    """SHAP -> the top features driving ONE pitcher's risk score.
    The returned feature names feed straight into the RAG layer's query."""
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    contribs = pd.Series(sv[row_idx], index=X.columns)
    top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(top_n)
    return list(top.index)

if __name__ == "__main__":
    season = build_season_table(build_features(pd.read_parquet(PARQUET_PATH)), load_surgeries())
    out = train_model(season)
    print("top features:", explain_prediction(out["model"], out["X_test"]))