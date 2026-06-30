import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
 
 
def plot_results(result, save_prefix="pitchguard"):
    """`result` is the dict from train_model:
    {model, X_test, y_test, proba, preds, threshold}."""
    y_test = result["y_test"]
    proba  = result["proba"]
    preds  = result["preds"]
    model  = result["model"]
    X_test = result["X_test"]
 
    base_rate = y_test.mean()
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fpr, tpr, _  = roc_curve(y_test, proba)
 
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
 
    # 1) precision-recall: the honest curve for rare events
    ax[0].plot(rec, prec, lw=2)
    ax[0].axhline(base_rate, ls="--", color="gray", label=f"random ({base_rate:.3f})")
    ax[0].set(xlabel="Recall (injuries caught)", ylabel="Precision (flags that are right)",
              title="Precision-Recall curve", xlim=(0, 1), ylim=(0, 1))
    ax[0].legend()
 
    # 2) ROC: ranking ability, independent of threshold
    ax[1].plot(fpr, tpr, lw=2)
    ax[1].plot([0, 1], [0, 1], ls="--", color="gray", label="random")
    ax[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
              title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    ax[1].legend()
 
    # 3) confusion matrix at the chosen threshold
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds), display_labels=["safe", "injured"]
    ).plot(ax=ax[2], colorbar=False, cmap="Blues")
    ax[2].set_title("Confusion matrix")
 
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_metrics.png", dpi=130)
 
    # 4) SHAP beeswarm: which features matter AND in which direction
    sv = shap.TreeExplainer(model).shap_values(X_test)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_test, show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_shap.png", dpi=130, bbox_inches="tight")
 
    plt.show()
    print(f"saved {save_prefix}_metrics.png and {save_prefix}_shap.png")
 
 
if __name__ == "__main__":
    import pandas as pd
    from config import PARQUET_PATH
    from features import build_features, build_season_table, load_surgeries
    from model import train_model
    season = build_season_table(
        build_features(pd.read_parquet(PARQUET_PATH)), load_surgeries())
    plot_results(train_model(season))