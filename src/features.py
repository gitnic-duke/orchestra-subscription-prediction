"""
Build the modeling table: one row per account, one column per feature.

The same function is used for train and test accounts, so both always get
exactly the same features. Every feature uses only data from seasons before
2014-15 (see TARGET_SEASON_START in config.py) to avoid leaking the future.

Features are organized in groups (one function per data source), so
experiments can turn whole groups on or off.

Patrons with no history in a table get NaN for that table's features
(e.g. a patron who never subscribed has no "last season subscribed").
Count-type features and yes/no flags are filled with 0 instead, since
"no rows" really means zero purchases.
"""
import pandas as pd

from src.config import TARGET_SEASON_START
from src import load_data

LAST_SEASON = TARGET_SEASON_START - 1   # 2013 = the 2013-14 season


# ---------------------------------------------------------------------------
# Subscription history
# ---------------------------------------------------------------------------

def last_streak_length(years):
    """
    Length of the unbroken run of seasons ending at the patron's most recent
    subscription. E.g. seasons [2005, 2010, 2011, 2012] -> 3.
    """
    years = sorted(set(years))
    streak = 1
    for i in range(len(years) - 1, 0, -1):
        if years[i] - years[i - 1] == 1:
            streak += 1
        else:
            break
    return streak


def subscription_features(subs):
    """Summarize each patron's past season subscriptions."""
    subs = subs[subs["season_year"] < TARGET_SEASON_START]
    subs = subs.sort_values(["account.id", "season_year"])

    # Group the 10 package names into 4 types: 'Quartet CC' -> 'Quartet',
    # 'Trio A' -> 'Trio', 'Full upgrade' -> 'Full'. The rare variants have
    # too few rows to learn from on their own.
    subs = subs.assign(package_type=subs["package"].str.split().str[0])

    by_account = subs.groupby("account.id")
    years = by_account["season_year"]

    # --- How recently and how often did they subscribe? ---
    feats = pd.DataFrame({
        "sub_last_season": years.max(),
        "sub_first_season": years.min(),
        "sub_n_seasons": years.nunique(),
        "sub_n_last5": years.apply(lambda y: (y > LAST_SEASON - 5).sum()),
        "sub_last_streak": years.apply(last_streak_length),
    })
    feats["sub_seasons_since_last"] = TARGET_SEASON_START - feats["sub_last_season"]
    feats["sub_2013"] = (feats["sub_last_season"] == LAST_SEASON).astype(int)
    feats["sub_2012"] = years.apply(lambda y: int((y == LAST_SEASON - 1).any()))

    # --- What did their most recent subscription look like? ---
    latest = by_account.tail(1).set_index("account.id")
    feats["sub_last_seats"] = latest["no.seats"]
    feats["sub_last_price_level"] = latest["price.level"]
    feats["sub_last_tier"] = latest["subscription_tier"]
    # Categorical columns use the categories of the whole file, so train and
    # test always get the same category codes.
    feats["sub_last_package"] = latest["package_type"].astype(
        pd.CategoricalDtype(sorted(subs["package_type"].dropna().unique())))
    feats["sub_last_location"] = latest["location"].astype(
        pd.CategoricalDtype(sorted(subs["location"].dropna().unique())))

    # --- Did they scale up or down compared to the subscription before? ---
    previous = by_account.nth(-2).set_index("account.id")   # NaN if only 1 season
    feats["sub_seats_change"] = latest["no.seats"] - previous["no.seats"]
    feats["sub_tier_change"] = latest["subscription_tier"] - previous["subscription_tier"]

    return feats


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------

# Filled with 0 when a patron has no rows in that table
ZERO_FILL_COLUMNS = ["sub_n_seasons", "sub_n_last5", "sub_last_streak",
                     "sub_2013", "sub_2012"]

ALL_GROUPS = ["subscriptions"]


def build_features(account_ids, groups=ALL_GROUPS):
    """
    Return a DataFrame indexed by account.id with the requested feature groups.

    account_ids: list/Series of account IDs (train or test), in the order
                 you want the rows returned.
    groups:      which feature groups to include (see ALL_GROUPS).
    """
    X = pd.DataFrame(index=pd.Index(account_ids, name="account.id"))

    if "subscriptions" in groups:
        X = X.join(subscription_features(load_data.load_subscriptions()))

    fill_cols = [c for c in ZERO_FILL_COLUMNS if c in X.columns]
    X[fill_cols] = X[fill_cols].fillna(0)
    return X
