"""
Run named experiments: build features, cross-validate a model, and log the
result to results/cv_results.csv.

Usage (from the repo root):
    python experiments.py --list                  # show experiment names
    python experiments.py <name>                  # run one experiment
    python experiments.py <name_a> <name_b>       # run both and compare them

Because every experiment uses the same CV folds, two experiments can be
compared fold by fold ("paired"). That is more reliable than comparing the
two means, since some folds are simply harder than others for every model.
"""
import sys

from src import load_data, models
from src.evaluate import cv_auc, log_result
from src.features import ALL_GROUPS, build_features

BASIC_SUB_COLUMNS = ["sub_last_season", "sub_first_season",
                     "sub_n_seasons", "sub_seasons_since_last"]

# Each experiment = which feature groups, (optionally) which columns, and which model
EXPERIMENTS = {
    # --- Step 1: subscription features, default model ---
    # Same 4 features the baseline rule uses -> isolates what the model adds
    "hgb_sub_basic": {
        "groups": ["subscriptions"], "columns": BASIC_SUB_COLUMNS,
        "model": models.gradient_boosting_default,
        "notes": "HGB defaults, 4 basic subscription features",
    },
    "hgb_sub_all": {
        "groups": ["subscriptions"], "columns": None,
        "model": models.gradient_boosting_default,
        "notes": "HGB defaults, all subscription features",
    },
    # --- Constrained (regularized) model from here on ---
    "hgb_reg_sub_basic": {
        "groups": ["subscriptions"], "columns": BASIC_SUB_COLUMNS,
        "model": models.gradient_boosting,
        "notes": "HGB regularized, 4 basic subscription features",
    },
    "hgb_reg_sub_all": {
        "groups": ["subscriptions"], "columns": None,
        "model": models.gradient_boosting,
        "notes": "HGB regularized, all subscription features",
    },
    # --- Step 2: add ticket features ---
    "hgb_reg_sub_tix": {
        "groups": ["subscriptions", "tickets"], "columns": None,
        "model": models.gradient_boosting,
        "notes": "HGB regularized, subscription + ticket features",
    },
    # --- Step 3: add donation/account features, raw vs. leak-cleaned ---
    "hgb_reg_sub_tix_don_raw": {
        "groups": ["subscriptions", "tickets", "donations_raw"], "columns": None,
        "model": models.gradient_boosting,
        "notes": "HGB regularized, sub + tix + donations as given (incl. 2014 first donations)",
    },
    "hgb_reg_sub_tix_don": {
        "groups": ["subscriptions", "tickets", "donations"], "columns": None,
        "model": models.gradient_boosting,
        "notes": "HGB regularized, sub + tix + donations with 2014 first-time donors removed",
    },
    # --- Step 4: add geography features (on top of the raw-donations set) ---
    "hgb_reg_sub_tix_don_raw_geo": {
        "groups": ["subscriptions", "tickets", "donations_raw", "geography"], "columns": None,
        "model": models.gradient_boosting,
        "notes": "HGB regularized, sub + tix + raw donations + geography",
    },
    # --- Step 5: compare algorithms on the chosen feature set (ALL_GROUPS) ---
    "logreg": {
        "groups": ALL_GROUPS, "columns": None,
        "model": models.logistic_regression,
        "notes": "Logistic regression (C=1), final feature set",
    },
    "random_forest": {
        "groups": ALL_GROUPS, "columns": None,
        "model": models.random_forest,
        "notes": "Random forest (500 trees, min leaf 5), final feature set",
    },
    "hgb_reg": {
        "groups": ALL_GROUPS, "columns": None,
        "model": models.gradient_boosting,
        "notes": "HGB regularized, final feature set",
    },
}


def run(name):
    """Run one experiment, log it, and return its 15 fold scores."""
    exp = EXPERIMENTS[name]
    train = load_data.load_train()
    X = build_features(train["account.id"], groups=exp["groups"])
    if exp["columns"] is not None:
        X = X[exp["columns"]]

    print(f"{name}: {X.shape[1]} features, {len(X)} accounts")
    fold_aucs = cv_auc(exp["model"](), X, train["label"])
    log_result(name, fold_aucs, notes=exp["notes"])
    return fold_aucs


def compare(name_a, name_b):
    """Run two experiments and print their fold-by-fold (paired) difference."""
    a, b = run(name_a), run(name_b)
    diff = b - a
    print(f"\n{name_b} minus {name_a}: mean {diff.mean():+.4f}, "
          f"better in {(diff > 0).sum()}/{len(diff)} folds")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1 and args[0] != "--list":
        run(args[0])
    elif len(args) == 2:
        compare(*args)
    else:
        print("Experiments:", ", ".join(EXPERIMENTS))
