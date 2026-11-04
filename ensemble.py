"""
Test whether averaging tuned gradient boosting with tuned logistic regression
beats gradient boosting alone (8 selected features, same 15 CV folds).

Tries several weights and both averaging methods (ranks vs probabilities),
then re-tests the best ensemble on 3 fresh sets of folds, since choosing the
best weight on the same folds can flatter it.

Run from the repo root:  python ensemble.py
"""
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src import load_data, models
from src.evaluate import cv_auc, log_result
from src.features import SELECTED_FEATURES, build_features

WEIGHTS_FOR_BOOSTING = [0.9, 0.7, 0.5, 0.3]   # rest goes to logistic regression


def paired(scores, reference):
    diff = scores - reference
    return f"{diff.mean():+.4f} vs boosting alone, better in {(diff > 0).sum()}/15 folds"


def main():
    train = load_data.load_train()
    y = train["label"]
    X = build_features(train["account.id"])[SELECTED_FEATURES]

    boosting = cv_auc(models.gradient_boosting_tuned(), X, y)
    logreg = cv_auc(models.logistic_regression_tuned(), X, y)
    print(f"boosting alone: {boosting.mean():.4f}   logistic regression alone: {logreg.mean():.4f}")

    print("\n=== Weighted averages ===")
    best_name, best_mean, best_settings = None, 0, None
    for method in ["rank", "probability"]:
        for w in WEIGHTS_FOR_BOOSTING:
            name = f"ensemble_{method}_{int(w * 100)}_{int(round((1 - w) * 100))}"
            scores = cv_auc(models.ensemble(w, method), X, y)
            log_result(name, scores, notes=f"{method} average, {w:.0%} boosting / {1 - w:.0%} logreg, 8 features")
            print(f"   {paired(scores, boosting)}")
            if scores.mean() > best_mean:
                best_name, best_mean, best_settings = name, scores.mean(), (w, method)

    print(f"\n=== Re-test of the best ensemble ({best_name}) on fresh folds ===")
    for seed in [1, 2, 3]:
        folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        alone = cross_val_score(models.gradient_boosting_tuned(), X, y, cv=folds, scoring="roc_auc")
        ens = cross_val_score(models.ensemble(*best_settings), X, y, cv=folds, scoring="roc_auc")
        diff = ens - alone
        print(f"seed {seed}: boosting {alone.mean():.4f}, ensemble {ens.mean():.4f}, "
              f"diff {diff.mean():+.4f}, ensemble better in {(diff > 0).sum()}/15 folds")


if __name__ == "__main__":
    main()
