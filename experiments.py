"""
Run one named experiment: build features, cross-validate a model, and log
the result to results/cv_results.csv.

Usage (from the repo root):
    python experiments.py <experiment_name>
    python experiments.py --list
"""
import sys

from src import load_data, models
from src.evaluate import cv_auc, log_result
from src.features import build_features

# Each experiment = which feature groups, (optionally) which columns, and which model
EXPERIMENTS = {
    # Same 4 features the baseline rule uses -> isolates what the model adds
    "hgb_sub_basic": {
        "groups": ["subscriptions"],
        "columns": ["sub_last_season", "sub_first_season",
                    "sub_n_seasons", "sub_seasons_since_last"],
        "model": models.gradient_boosting,
        "notes": "HGB defaults, 4 basic subscription features",
    },
    # All subscription-history features
    "hgb_sub_all": {
        "groups": ["subscriptions"],
        "columns": None,
        "model": models.gradient_boosting,
        "notes": "HGB defaults, all subscription features",
    },
}


def run(name):
    exp = EXPERIMENTS[name]
    train = load_data.load_train()
    X = build_features(train["account.id"], groups=exp["groups"])
    if exp["columns"] is not None:
        X = X[exp["columns"]]

    print(f"{name}: {X.shape[1]} features, {len(X)} accounts")
    fold_aucs = cv_auc(exp["model"](), X, train["label"])
    log_result(name, fold_aucs, notes=exp["notes"])


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] == "--list":
        print("Experiments:", ", ".join(EXPERIMENTS))
    else:
        run(sys.argv[1])
