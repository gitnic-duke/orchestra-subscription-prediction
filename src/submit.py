"""
Write a Kaggle submission file in the exact format of sample_submission.csv.
"""
import pandas as pd

from src.config import DATA_DIR, SUBMISSION_DIR


def save_submission(account_ids, predictions, filename):
    """
    Save predicted probabilities as submissions/<filename>.

    Checks that we predict every test account exactly once, in the same
    order as sample_submission.csv, and that predictions are probabilities
    (AUROC needs soft scores, not 0/1 labels).
    """
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = pd.DataFrame({"ID": list(account_ids), "Predicted": list(predictions)})

    assert list(sub["ID"]) == list(sample["ID"]), "IDs/order differ from sample_submission"
    assert sub["Predicted"].between(0, 1).all(), "predictions must be in [0, 1]"
    assert sub["Predicted"].notna().all(), "predictions contain NaN"

    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / filename
    sub.to_csv(path, index=False)
    print(f"Saved {len(sub)} predictions to {path}")
    return path
