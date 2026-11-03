"""
Validation helpers.

We judge every idea by cross-validated AUROC on the training set rather than
by the public leaderboard. With only ~330 positive accounts in train (4.8%),
a single train/validation split is noisy, so we use 5-fold stratified CV
repeated 3 times (15 scores) and report the mean and standard deviation.
Stratifying keeps the 4.8% positive rate the same in every fold.
"""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src.config import RANDOM_SEED, RESULTS_DIR

N_SPLITS = 5
N_REPEATS = 3


def get_folds():
    """The CV splitter used everywhere, so all models see identical folds."""
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED
    )


def cv_auc(model, X, y):
    """Cross-validated AUROC scores (one per fold) for a scikit-learn model."""
    return cross_val_score(model, X, y, cv=get_folds(), scoring="roc_auc")


def cv_auc_of_fixed_scores(scores, y):
    """
    AUROC per fold for a hand-made score that needs no training (a heuristic).
    Evaluated on the same validation folds as the models, so it is comparable.
    """
    scores, y = np.asarray(scores), np.asarray(y)
    return np.array([
        roc_auc_score(y[val_idx], scores[val_idx])
        for _, val_idx in get_folds().split(scores.reshape(-1, 1), y)
    ])


def log_result(experiment, fold_aucs, notes=""):
    """
    Print the CV result and append it to results/cv_results.csv,
    so we keep a record of every experiment (useful for the writeup).
    """
    mean, std = np.mean(fold_aucs), np.std(fold_aucs)
    print(f"{experiment}: CV AUROC = {mean:.4f} (+/- {std:.4f})")

    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "cv_results.csv"
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": experiment,
        "cv_auc_mean": round(mean, 4),
        "cv_auc_std": round(std, 4),
        "notes": notes,
    }])
    row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
