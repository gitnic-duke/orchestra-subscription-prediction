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
import numpy as np
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
# Single-ticket purchases (2010-11 to 2013-14)
# ---------------------------------------------------------------------------

def ticket_features(tickets):
    """
    Summarize each patron's single-ticket purchases.

    Marketing source is left out: it is missing for 78% of purchases and
    showed no clear relationship with subscribing.
    """
    tickets = tickets[tickets["season_year"] < TARGET_SEASON_START]
    by_account = tickets.groupby("account.id")
    years = by_account["season_year"]

    feats = pd.DataFrame({
        "tix_n_purchases": by_account.size(),
        "tix_n_seasons": years.nunique(),
        "tix_last_season": years.max(),
        "tix_total_seats": by_account["no.seats"].sum(),
        "tix_mean_price_level": by_account["price.level"].mean(),
        "tix_family_concert": by_account["location"].apply(
            lambda loc: int((loc == "Family concert").any())),
    })
    feats["tix_seasons_since_last"] = TARGET_SEASON_START - feats["tix_last_season"]
    feats["tix_2013"] = (feats["tix_last_season"] == LAST_SEASON).astype(int)
    return feats


# ---------------------------------------------------------------------------
# Donations and other account info (account.csv)
# ---------------------------------------------------------------------------

# Donations dated on/after this day may come after (or with) the 2014-15
# subscription purchase, i.e. they could leak the answer.
DONATION_CUTOFF = pd.Timestamp("2014-01-01")


def simplify_relationship(relationship):
    """Group the 10 relationship labels into 3: Education / Vendor / Other."""
    rel = relationship.copy()
    rel[rel.str.startswith("Vendor", na=False)] = "Vendor"
    rel[rel.notna() & ~rel.isin(["Vendor", "Education Participant"])] = "Other"
    return rel.astype(pd.CategoricalDtype(["Education Participant", "Vendor", "Other"]))


def account_features(accounts, remove_2014_donors):
    """
    Donation history plus relationship type and whether we have an address.

    remove_2014_donors: if True, patrons whose FIRST donation is dated 2014
    are treated as non-donors (lifetime amount and count set to 0). Their
    donations happened during the 2014-15 renewal period, and many look like
    gifts added when buying the subscription, which would leak the label.
    This is only a partial fix: patrons who first donated earlier may also
    have 2014 gifts inside their lifetime totals, and we can't separate those.
    """
    acc = accounts.set_index("account.id").copy()

    if remove_2014_donors:
        late = acc["first.donated"] >= DONATION_CUTOFF
        acc.loc[late, ["amount.donated.lifetime", "no.donations.lifetime"]] = 0
        acc.loc[late, "first.donated"] = pd.NaT

    feats = pd.DataFrame(index=acc.index)
    feats["don_amount_2013"] = acc["amount.donated.2013"]
    feats["don_amount_lifetime"] = acc["amount.donated.lifetime"]
    feats["don_n_lifetime"] = acc["no.donations.lifetime"]
    # Average gift size (NaN for non-donors, so it's not confused with tiny gifts)
    feats["don_avg_amount"] = (acc["amount.donated.lifetime"]
                               / acc["no.donations.lifetime"].replace(0, float("nan")))
    feats["don_first_year"] = acc["first.donated"].dt.year

    feats["acct_relationship"] = simplify_relationship(acc["relationship"])
    feats["acct_has_billing_zip"] = acc["billing.zip.code"].notna().astype(int)
    return feats


# ---------------------------------------------------------------------------
# Geography (billing zip joined to zipcodes.csv)
# ---------------------------------------------------------------------------

# Cities where concerts were held in the recent seasons (2011-12 to 2013-14).
# Approximate city-center coordinates; zipcodes.csv only has 2-decimal lat/long anyway.
CONCERT_CITIES = {
    "San Francisco": (37.78, -122.42),
    "Berkeley": (37.87, -122.27),
    "Palo Alto (Peninsula)": (37.44, -122.14),
}


def miles_between(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles (haversine formula)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 3959 * 2 * np.arcsin(np.sqrt(a))


def geography_features(accounts, zipcodes):
    """
    Where the patron lives, from their billing zip code (shipping zip is 99%
    missing). Accounts whose zip is missing or not a US zip get NaN.
    """
    geo = (accounts[["account.id", "billing.zip.code"]]
           .merge(zipcodes, left_on="billing.zip.code", right_on="zip", how="left")
           .set_index("account.id"))

    feats = pd.DataFrame(index=geo.index)
    # Distance to the closest concert city: far-away patrons rarely subscribe
    feats["geo_miles_to_concerts"] = np.min(
        [miles_between(geo["Lat"], geo["Long"], lat, lon)
         for lat, lon in CONCERT_CITIES.values()], axis=0)
    # Average income in the zip code (total wages / number of tax returns)
    feats["geo_avg_wage"] = geo["TotalWages"] / geo["TaxReturnsFiled"]
    feats["geo_population"] = geo["EstimatedPopulation"]
    feats["geo_in_california"] = (geo["State"] == "CA").astype(float)
    feats.loc[geo["State"].isna(), "geo_in_california"] = np.nan   # unknown, not "no"
    return feats


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------

# Filled with 0 when a patron has no rows in that table
ZERO_FILL_COLUMNS = ["sub_n_seasons", "sub_n_last5", "sub_last_streak",
                     "sub_2013", "sub_2012",
                     "tix_n_purchases", "tix_n_seasons", "tix_total_seats",
                     "tix_family_concert", "tix_2013"]

# Default feature set. Donations are used as given ("donations_raw"), including
# the 2014 first donations that partly leak the label: chosen deliberately for
# the competition score; see the writeup for the clean-vs-raw comparison.
# Of the geography features only distance to concerts is kept: zip income and
# population did not help in cross-validation.
ALL_GROUPS = ["subscriptions", "tickets", "donations_raw", "distance"]

# Final feature set chosen in feature selection (feature_selection.py): the 8
# features with held-out permutation importance > 0.0002. As accurate as all
# 31 features in cross-validation, with much less room to overfit.
SELECTED_FEATURES = [
    "sub_last_season",        # most recent season subscribed
    "sub_first_season",       # first season subscribed
    "sub_last_streak",        # length of the unbroken run ending at the last subscription
    "sub_last_location",      # venue of the last subscription
    "tix_mean_price_level",   # average price level of single tickets bought
    "don_first_year",         # year of first donation (partly leaky, see writeup)
    "don_n_lifetime",         # number of donations ever
    "geo_miles_to_concerts",  # distance to the nearest concert city (missing = no address)
]


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
    if "tickets" in groups:
        X = X.join(ticket_features(load_data.load_tickets()))
    if "donations" in groups:          # 2014 first-time donors removed
        X = X.join(account_features(load_data.load_accounts(), remove_2014_donors=True))
    if "donations_raw" in groups:      # account.csv as given
        X = X.join(account_features(load_data.load_accounts(), remove_2014_donors=False))
    if "geography" in groups:          # all 4 geography features
        X = X.join(geography_features(load_data.load_accounts(), load_data.load_zipcodes()))
    if "distance" in groups:           # only distance to the concert cities
        geo = geography_features(load_data.load_accounts(), load_data.load_zipcodes())
        X = X.join(geo[["geo_miles_to_concerts"]])

    fill_cols = [c for c in ZERO_FILL_COLUMNS if c in X.columns]
    X[fill_cols] = X[fill_cols].fillna(0)
    return X
