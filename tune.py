"""
Hyperparameter tuning on the 8 selected features, using the same 15 CV folds
as every other experiment.

1. Gradient boosting: randomized search over 60 random combinations of
   settings (a full grid would be thousands of combinations).
2. Logistic regression: grid search over the regularization strength C.
3. Robustness check: picking the best of 60 candidates on the same folds can
   flatter the winner (some of its lead is luck). So the winner is re-tested
   against the current settings on 3 differently shuffled sets of folds.

Run from the repo root:  python tune.py
Saves every candidate's score to results/tuning_gradient_boosting.csv and
results/tuning_logistic_regression.csv
"""
import pandas as pd
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, cross_val_score)

from src import load_data, models
from src.config import RANDOM_SEED, RESULTS_DIR
from src.evaluate import get_folds
from src.features import SELECTED_FEATURES, build_features

# Values to try for each gradient-boosting setting
HGB_SEARCH_SPACE = {
    "learning_rate": [0.01, 0.02, 0.05, 0.1],   # size of each boosting step
    "max_iter": [100, 200, 400, 800],           # number of trees
    "max_depth": [2, 3, 4, 6],                  # depth of each tree
    "min_samples_leaf": [10, 20, 40, 80, 150],  # min patrons per leaf
    "l2_regularization": [0.0, 0.1, 1.0, 10.0], # shrinks leaf values
    "max_features": [0.5, 0.75, 1.0],           # share of features per split
}
N_CANDIDATES = 60

LOGREG_C_VALUES = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]  # smaller C = stronger penalty


def search_results_table(search, param_names):
    """Tidy table of every candidate: settings, mean and std of CV AUROC."""
    results = pd.DataFrame(search.cv_results_)
    table = results[[f"param_{p}" for p in param_names]
                    + ["mean_test_score", "std_test_score"]]
    table.columns = param_names + ["cv_auc_mean", "cv_auc_std"]
    return table.sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def tune_gradient_boosting(X, y):
    print(f"=== Gradient boosting: {N_CANDIDATES} random combinations x 15 folds ===")
    search = RandomizedSearchCV(
        models.gradient_boosting(), HGB_SEARCH_SPACE, n_iter=N_CANDIDATES,
        scoring="roc_auc", cv=get_folds(), random_state=RANDOM_SEED,
        n_jobs=-1, refit=False)
    search.fit(X, y)
    table = search_results_table(search, list(HGB_SEARCH_SPACE))
    table.to_csv(RESULTS_DIR / "tuning_gradient_boosting.csv", index=False)
    print(table.head(10).round(4).to_string())
    return search.best_params_


def tune_logistic_regression(X, y):
    print("\n=== Logistic regression: C values x 15 folds ===")
    search = GridSearchCV(
        models.logistic_regression(), {"logisticregression__C": LOGREG_C_VALUES},
        scoring="roc_auc", cv=get_folds(), n_jobs=-1, refit=False)
    search.fit(X, y)
    table = search_results_table(search, ["logisticregression__C"]).rename(
        columns={"logisticregression__C": "C"})
    table.to_csv(RESULTS_DIR / "tuning_logistic_regression.csv", index=False)
    print(table.round(4).to_string())
    return search.best_params_


def robustness_check(X, y, best_params):
    """Winner vs. current settings on 3 fresh sets of folds (paired)."""
    print("\n=== Robustness check: tuned vs current settings on new folds ===")
    for seed in [1, 2, 3]:
        folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        current = cross_val_score(models.gradient_boosting(), X, y,
                                  cv=folds, scoring="roc_auc")
        tuned = cross_val_score(models.gradient_boosting().set_params(**best_params),
                                X, y, cv=folds, scoring="roc_auc")
        diff = tuned - current
        print(f"seed {seed}: current {current.mean():.4f}, tuned {tuned.mean():.4f}, "
              f"diff {diff.mean():+.4f}, tuned better in {(diff > 0).sum()}/15 folds")


def main():
    train = load_data.load_train()
    y = train["label"]
    X = build_features(train["account.id"])[SELECTED_FEATURES]
    RESULTS_DIR.mkdir(exist_ok=True)

    best_hgb = tune_gradient_boosting(X, y)
    print("\nBest gradient-boosting settings:", best_hgb)
    tune_logistic_regression(X, y)
    robustness_check(X, y, best_hgb)


if __name__ == "__main__":
    main()
