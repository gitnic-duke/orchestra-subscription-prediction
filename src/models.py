"""
Model definitions. Each function returns a fresh, untrained scikit-learn model,
so every experiment builds its model the same way.
"""
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import RANDOM_SEED


def gradient_boosting_default():
    """
    Gradient-boosted decision trees (scikit-learn's histogram version,
    similar to LightGBM) with scikit-learn's default hyperparameters.

    Why this model:
    - handles missing values natively (most patrons have no history in
      some tables), so no imputation is needed
    - handles categorical columns directly (categorical_features="from_dtype")
    - captures non-linear patterns, e.g. "never subscribed" vs
      "lapsed 5+ seasons ago" having different renewal rates
    Used in step 1; it overfit slightly.
    """
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=RANDOM_SEED,
    )


def gradient_boosting():
    """
    Same model, constrained to overfit less (we only have 332 positives).
    Working settings used for comparing feature sets; tuned properly later.

    Changes from the defaults (default value in brackets):
    - max_depth=3 [no limit, 31 leaves]: shallower trees, simpler rules
    - learning_rate=0.05 [0.1] with max_iter=200 [100]: smaller steps,
      more trees, so each tree has less influence
    - min_samples_leaf=40 [20]: every rule must cover more patrons
    - l2_regularization=1.0 [0]: shrinks leaf values
    """
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=RANDOM_SEED,
    )
