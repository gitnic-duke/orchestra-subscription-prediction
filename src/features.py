"""
Build the modeling table: one row per account, one column per feature.

The same function is used for train and test accounts, so both always get
exactly the same features. Every feature uses only data from seasons before
2014-15 (see TARGET_SEASON_START in config.py) to avoid leaking the future.

Patrons with no history in a table get NaN for that table's features
(e.g. a patron who never subscribed has no "last season subscribed").
Count-type features are filled with 0 instead, since "no rows" really means
zero purchases.
"""
import pandas as pd

from src.config import TARGET_SEASON_START
from src import load_data


# ---------------------------------------------------------------------------
# Subscription history
# ---------------------------------------------------------------------------

def subscription_features(subs):
    """Summarize each patron's past season subscriptions."""
    subs = subs[subs["season_year"] < TARGET_SEASON_START]
    by_account = subs.groupby("account.id")["season_year"]

    feats = pd.DataFrame({
        "sub_last_season": by_account.max(),
        "sub_first_season": by_account.min(),
        "sub_n_seasons": by_account.nunique(),
    })
    # How many seasons ago did they last subscribe? (1 = subscribed in 2013-14)
    feats["sub_seasons_since_last"] = TARGET_SEASON_START - feats["sub_last_season"]
    return feats


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------

COUNT_COLUMNS = ["sub_n_seasons"]   # filled with 0 when a patron has no rows


def build_features(account_ids):
    """
    Return a DataFrame indexed by account.id with all features.

    account_ids: list/Series of account IDs (train or test), in the order
    you want the rows returned.
    """
    subs = load_data.load_subscriptions()

    X = pd.DataFrame(index=pd.Index(account_ids, name="account.id"))
    X = X.join(subscription_features(subs))

    X[COUNT_COLUMNS] = X[COUNT_COLUMNS].fillna(0)
    return X
