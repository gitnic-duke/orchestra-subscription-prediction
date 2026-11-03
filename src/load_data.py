"""
Load and clean the raw Kaggle files.

Each function returns one tidy pandas DataFrame. Cleaning is kept here so
that every other script works with the same, consistent data.
"""
import pandas as pd

from src.config import DATA_DIR


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def season_start_year(season):
    """Turn a season label like '2013-2014' into its starting year (2013)."""
    return season.str[:4].astype(int)


def clean_zip(zip_series):
    """
    Standardize US zip codes to 5-character strings.

    - '94501-3315' -> '94501'  (drop the +4 extension)
    - '2116'       -> '02116'  (leading zero was lost, e.g. Boston)
    Foreign postal codes (e.g. Canadian 'T8N 3Z9') simply won't match the
    zip code table later, which is fine.
    """
    z = zip_series.astype("string").str.strip().str[:5]
    is_four_digits = z.str.fullmatch(r"\d{4}", na=False)
    z[is_four_digits] = z[is_four_digits].str.zfill(5)
    return z


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def load_train():
    """Training accounts with the target: 1 = bought a 2014-15 subscription."""
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test():
    """Test accounts we must predict. Renamed to match train ('account.id')."""
    return pd.read_csv(DATA_DIR / "test.csv").rename(columns={"ID": "account.id"})


# ---------------------------------------------------------------------------
# Patron-level data
# ---------------------------------------------------------------------------

def load_accounts():
    """
    One row per patron: location and donation history.

    The file is not UTF-8 (it contains accented city names), so we read it
    as latin-1.
    """
    acc = pd.read_csv(
        DATA_DIR / "account.csv",
        encoding="latin1",
        dtype={"shipping.zip.code": str, "billing.zip.code": str},
    )
    acc["billing.zip.code"] = clean_zip(acc["billing.zip.code"])
    acc["shipping.zip.code"] = clean_zip(acc["shipping.zip.code"])
    acc["first.donated"] = pd.to_datetime(acc["first.donated"], errors="coerce")
    return acc


def load_subscriptions():
    """
    One row per (patron, season) subscription, 1993-94 through 2013-14.

    Missing price.level / section are left as NaN; the models we use
    (gradient boosting) handle missing values natively.
    """
    subs = pd.read_csv(DATA_DIR / "subscriptions.csv")
    subs["season_year"] = season_start_year(subs["season"])
    subs["multiple.subs"] = (subs["multiple.subs"] == "yes").astype(int)
    return subs


def load_tickets():
    """
    Single-ticket purchases, 2010-11 through 2013-14.

    price.level mixes numbers with labels like 'Adult', 'Youth', 'GA' and
    '4.0'. We keep the numeric levels and set the labels to NaN.
    """
    tickets = pd.read_csv(DATA_DIR / "tickets_all.csv")
    tickets["season_year"] = season_start_year(tickets["season"])
    tickets["price.level"] = pd.to_numeric(tickets["price.level"], errors="coerce")
    tickets["multiple.tickets"] = (tickets["multiple.tickets"] == "yes").astype(int)
    return tickets


def load_zipcodes():
    """
    Location and demographic info per US zip code.

    Note: this file uses old Mac line endings (\\r), so `wc -l` shows 0 lines,
    but pandas reads it fine.
    """
    zips = pd.read_csv(DATA_DIR / "zipcodes.csv")
    zips["zip"] = zips["Zipcode"].astype(str).str.zfill(5)
    keep = ["zip", "State", "Lat", "Long",
            "TaxReturnsFiled", "EstimatedPopulation", "TotalWages"]
    return zips[keep]


def load_concerts():
    """Past concerts (2010-11 to 2013-14), one row per performance."""
    concerts = pd.read_csv(DATA_DIR / "concerts.csv", encoding="utf-8-sig")
    concerts["season_year"] = season_start_year(concerts["season"])
    return concerts


def load_concerts_2014():
    """The planned 2014-15 concert sets."""
    return pd.read_csv(DATA_DIR / "concerts_2014-15.csv",
                       encoding="latin1").rename(columns=lambda c: c.lstrip("ï»¿"))
