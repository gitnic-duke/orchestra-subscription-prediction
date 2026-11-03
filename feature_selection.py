"""
Feature selection for the gradient-boosting model, in three parts:

1. Group ablation: remove one feature group at a time and measure how much
   CV AUROC drops. Shows how much each data source contributes.
2. Permutation importance on held-out folds: for each CV fold, fit on the
   training part, then shuffle one feature at a time in the validation part
   and measure how much AUROC drops. A feature whose shuffling doesn't hurt
   isn't being used in a way that generalizes.
3. Drop the low-importance features and check CV again.

Run from the repo root:  python feature_selection.py
Saves results/feature_importance.csv and results/figures/feature_importance.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src import load_data, models
from src.config import RANDOM_SEED, RESULTS_DIR
from src.evaluate import cv_auc, get_folds, log_result
from src.features import ALL_GROUPS, build_features

BLUE, GRAY_TEXT = "#2a78d6", "#52514e"


def paired_summary(scores, reference):
    """Mean fold-by-fold difference vs. a reference, and # folds that improved."""
    diff = scores - reference
    return f"{diff.mean():+.4f} vs full, better in {(diff > 0).sum()}/{len(diff)} folds"


def group_ablation(account_ids, y, full_scores):
    """Remove each feature group in turn and re-run CV."""
    print("\n=== 1. Group ablation (remove one group at a time) ===")
    for group in ALL_GROUPS:
        kept = [g for g in ALL_GROUPS if g != group]
        X = build_features(account_ids, groups=kept)
        scores = cv_auc(models.gradient_boosting(), X, y)
        log_result(f"ablation_without_{group}", scores,
                   notes=f"HGB regularized, all groups except {group}")
        print(f"   {paired_summary(scores, full_scores)}")


def heldout_permutation_importance(X, y):
    """
    Average permutation importance over the 15 CV validation folds.
    Measured on data the model did NOT train on, so it reflects what
    actually generalizes (importance on training data can be misleading).
    """
    importances = []
    for train_idx, val_idx in get_folds().split(X, y):
        model = models.gradient_boosting().fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            model, X.iloc[val_idx], y.iloc[val_idx],
            scoring="roc_auc", n_repeats=5, random_state=RANDOM_SEED)
        importances.append(result.importances_mean)
    imp = pd.DataFrame(importances, columns=X.columns)
    return pd.DataFrame({"importance_mean": imp.mean(), "importance_std": imp.std()}
                        ).sort_values("importance_mean", ascending=False)


def plot_importance(importance, path):
    """Horizontal bar chart of held-out permutation importance."""
    imp = importance["importance_mean"].iloc[::-1]          # largest at the top
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=0.001)                # one feature dominates
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=GRAY_TEXT, labelsize=8)
    ax.set_xlabel("Drop in validation AUROC when the feature is shuffled (symlog scale)",
                  color=GRAY_TEXT)
    ax.set_title("Held-out permutation importance (gradient boosting)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    train = load_data.load_train()
    y = train["label"]
    X = build_features(train["account.id"])
    full_scores = cv_auc(models.gradient_boosting(), X, y)
    print(f"Full feature set: {X.shape[1]} features, CV AUROC {full_scores.mean():.4f}")

    # 1. How much does each data source contribute?
    group_ablation(train["account.id"], y, full_scores)

    # 2. Which individual features matter?
    print("\n=== 2. Held-out permutation importance ===")
    importance = heldout_permutation_importance(X, y)
    print(importance.round(4).to_string())
    RESULTS_DIR.mkdir(exist_ok=True)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv")
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    plot_importance(importance, RESULTS_DIR / "figures" / "feature_importance.png")

    # 3. Does dropping the weak features hurt?
    print("\n=== 3. Drop low-importance features ===")
    for threshold in [0.0, 0.0002, 0.0005]:
        keep = importance.index[importance["importance_mean"] > threshold]
        scores = cv_auc(models.gradient_boosting(), X[keep], y)
        log_result(f"select_importance_gt_{threshold}", scores,
                   notes=f"HGB regularized, {len(keep)} features with importance > {threshold}")
        print(f"   {len(keep)} features kept; {paired_summary(scores, full_scores)}")


if __name__ == "__main__":
    main()
