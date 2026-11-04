"""
Model definitions. Each function returns a fresh, untrained scikit-learn model,
so every experiment builds its model the same way.

Gradient boosting works on the feature table directly. Logistic regression
and random forest need some preprocessing first (missing values, categorical
columns), so they are wrapped in scikit-learn Pipelines. Because the
preprocessing sits inside the pipeline, it is re-fit on each CV training
fold, so nothing leaks from the validation fold.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

from src.config import RANDOM_SEED

# Column groups, picked by name/type so the models work with any feature set
MONEY_COLUMNS = make_column_selector(pattern=r"^don_(?:amount|avg)")   # very skewed $
CATEGORICAL_COLUMNS = make_column_selector(dtype_include="category")
OTHER_NUMERIC_COLUMNS = make_column_selector(pattern=r"^(?!don_(?:amount|avg))",
                                             dtype_exclude="category")


def logistic_regression(C=1.0):
    """
    Regularized logistic regression: a simple linear baseline model.

    Preprocessing it needs:
    - money columns: log(1 + x), since lifetime donations range from $0 to $1.2M
      and a linear model would otherwise be dominated by a few huge donors
    - missing values: filled with the median, plus a 0/1 "was missing" column,
      so "never subscribed" still gets its own weight
    - all numeric columns standardized (mean 0, std 1) so the L2 penalty
      treats them equally
    - categorical columns: one-hot encoded (missing becomes its own category)
    """
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                            StandardScaler())
    money = make_pipeline(FunctionTransformer(np.log1p), numeric)
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"))
    preprocess = ColumnTransformer([
        ("money", money, MONEY_COLUMNS),
        ("numeric", numeric, OTHER_NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ])
    return make_pipeline(preprocess, LogisticRegression(C=C, max_iter=5000))


def logistic_regression_tuned():
    """
    Logistic regression with the regularization strength chosen by grid search
    (tune.py): C = 3, i.e. a weak penalty. CV AUROC on the 8 selected features:
    0.9785 (vs 0.9778 at the default C = 1).
    """
    return logistic_regression(C=3.0)


def random_forest():
    """
    Random forest: many deep trees on random subsets of rows and features,
    averaged. A different way of combining trees than boosting.

    Missing values are handled natively by scikit-learn's random forest.
    Categorical columns are converted to integer codes (missing = -1).
    Reasonable settings, not tuned: 500 trees, at least 5 patrons per leaf,
    sqrt(#features) candidate features per split.
    """
    categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 encoded_missing_value=-1)
    preprocess = ColumnTransformer(
        [("categorical", categorical, CATEGORICAL_COLUMNS)],
        remainder="passthrough")
    return make_pipeline(preprocess, RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_SEED))


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


def gradient_boosting_tuned():
    """
    Gradient boosting with the settings chosen by randomized search over 60
    combinations (tune.py), on the 8 selected features. CV AUROC 0.9805
    (vs 0.9797 for the hand-picked settings above).

    - max_depth=2: very shallow trees (each captures at most a pairwise
      interaction); all of the top 3 settings used depth 2
    - learning_rate=0.05 with max_iter=400 trees
    - min_samples_leaf=20
    - l2_regularization=10: strong shrinkage of leaf values
    - max_features=0.75: each split considers a random 75% of the features,
      which adds variety between trees (like a random forest)
    """
    return HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=10.0,
        max_features=0.75,
        categorical_features="from_dtype",
        random_state=RANDOM_SEED,
    )


class AverageEnsemble(ClassifierMixin, BaseEstimator):
    """
    Combine several models by a weighted average of their predictions.

    method="probability" (default): weighted average of predicted probabilities.
    method="rank": each model's scores are first turned into ranks (scaled to
    0-1, computed within the set of patrons being predicted), then averaged.

    We tested both (ensemble.py). Probability averaging won: ~87% of patrons
    have a boosting probability under 1% and almost none of them subscribe.
    Ranks spread those patrons evenly from 0 to 0.87, so tiny, meaningless
    differences among near-certain non-subscribers count as much as the real
    gap between, say, 0.3% and 30%. Averaging probabilities keeps that gap.
    """

    def __init__(self, estimators, weights, method="probability"):
        self.estimators = estimators
        self.weights = weights
        self.method = method

    def fit(self, X, y):
        self.fitted_ = [clone(est).fit(X, y) for est in self.estimators]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        scores = []
        for model in self.fitted_:
            p = model.predict_proba(X)[:, 1]
            if self.method == "rank":
                p = rankdata(p) / len(p)          # ranks scaled to (0, 1]
            scores.append(p)
        combined = np.average(scores, axis=0, weights=self.weights)
        return np.column_stack([1 - combined, combined])


def ensemble(weight_boosting=0.5, method="probability"):
    """
    Tuned gradient boosting + tuned logistic regression, weighted average.
    Best in ensemble.py: probability average, 50/50 -> CV AUROC 0.9812
    (boosting alone 0.9805; better on all 4 sets of folds tested).
    """
    return AverageEnsemble(
        estimators=[gradient_boosting_tuned(), logistic_regression_tuned()],
        weights=[weight_boosting, 1 - weight_boosting],
        method=method)
