"""
Baseline: a hand-made rule with no machine learning.

Idea: patrons who subscribed recently, and for many seasons, are the most
likely to subscribe again. We rank patrons by
    1) the most recent season they subscribed, then
    2) the number of seasons they subscribed (as a tie-breaker).
Patrons who never subscribed get the lowest score.

This gives us (a) a score any model must beat and (b) a first Kaggle
submission that confirms our file format is correct.

Run from the repo root:  python baseline.py
"""
from src import load_data
from src.evaluate import cv_auc_of_fixed_scores, log_result
from src.features import build_features
from src.submit import save_submission


def recency_score(X):
    """Higher = subscribed more recently (and more often). Scaled to [0, 1]."""
    # e.g. last subscribed 2013-14 for 10 seasons -> 2013 * 100 + 10 = 201310
    raw = (X["sub_last_season"] * 100 + X["sub_n_seasons"]).fillna(0)
    return (raw - raw.min()) / (raw.max() - raw.min())


def main():
    train = load_data.load_train()
    test = load_data.load_test()

    # Evaluate the rule on the same CV folds we will use for models
    X_train = build_features(train["account.id"])
    fold_aucs = cv_auc_of_fixed_scores(recency_score(X_train), train["label"])
    log_result("baseline_recency_rule", fold_aucs,
               notes="rank by last season subscribed, then # seasons")

    # Score the test accounts and write the submission
    X_test = build_features(test["account.id"])
    save_submission(test["account.id"], recency_score(X_test), "baseline_recency_rule.csv")


if __name__ == "__main__":
    main()
