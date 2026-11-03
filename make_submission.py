"""
Train an experiment's model on ALL training accounts and predict the test set.

Cross-validation (experiments.py) is only for choosing features and models.
For the submission we refit the chosen setup on all 6,941 training accounts,
so the final model learns from every labeled patron.

Usage (from the repo root):
    python make_submission.py <experiment_name>
Writes submissions/<experiment_name>.csv in Kaggle's format.
"""
import sys

from experiments import EXPERIMENTS
from src import load_data
from src.features import build_features
from src.submit import save_submission


def make_submission(name):
    exp = EXPERIMENTS[name]
    train, test = load_data.load_train(), load_data.load_test()

    # Same feature-building code for train and test
    X_train = build_features(train["account.id"], groups=exp["groups"])
    X_test = build_features(test["account.id"], groups=exp["groups"])
    if exp["columns"] is not None:
        X_train, X_test = X_train[exp["columns"]], X_test[exp["columns"]]

    model = exp["model"]().fit(X_train, train["label"])

    # Probability of class 1 (subscribes). AUROC needs these soft scores.
    predictions = model.predict_proba(X_test)[:, 1]
    print(f"{name}: trained on {len(X_train)} accounts, {X_train.shape[1]} features; "
          f"mean predicted probability {predictions.mean():.3f} "
          f"(train positive rate {train['label'].mean():.3f})")
    return save_submission(test["account.id"], predictions, f"{name}.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_submission.py <experiment_name>")
        print("Experiments:", ", ".join(EXPERIMENTS))
    else:
        make_submission(sys.argv[1])
