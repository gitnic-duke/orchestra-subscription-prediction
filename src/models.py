"""
Model definitions. Each function returns a fresh, untrained scikit-learn model,
so every experiment builds its model the same way.
"""
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import RANDOM_SEED


def gradient_boosting():
    """
    Gradient-boosted decision trees (scikit-learn's histogram version,
    similar to LightGBM).

    Why this model first:
    - handles missing values natively (most patrons have no history in
      some tables), so no imputation is needed
    - handles categorical columns directly (categorical_features="from_dtype")
    - captures non-linear patterns, e.g. "never subscribed" vs
      "lapsed 5+ seasons ago" having different renewal rates
    Default hyperparameters for now; tuned in a later step.
    """
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=RANDOM_SEED,
    )
