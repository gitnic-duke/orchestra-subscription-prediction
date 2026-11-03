"""
Exploratory data analysis: key facts about the data, printed to the console,
plus two figures saved to results/figures/.

Run from the repo root:  python eda.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from src import load_data
from src.config import RESULTS_DIR
from src.features import build_features

BLUE, ORANGE = "#2a78d6", "#eb6834"   # colorblind-safe pair
GRAY_TEXT = "#52514e"


def style_axes(ax):
    """Light, unobtrusive axes so the data stands out."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors=GRAY_TEXT)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def print_overview(train, test, accounts, subs, tickets):
    print("=== Size and target ===")
    print(f"train: {len(train)} accounts, test: {len(test)} accounts")
    print(f"positive rate in train: {train['label'].mean():.3%} "
          f"({train['label'].sum()} subscribers)")
    print(f"train/test overlap: {len(set(train['account.id']) & set(test['account.id']))}")

    print("\n=== How many accounts have any history? ===")
    for name, ids in [("train", train["account.id"]), ("test", test["account.id"])]:
        print(f"{name}: subscriptions {ids.isin(subs['account.id']).mean():.1%}, "
              f"tickets {ids.isin(tickets['account.id']).mean():.1%}, "
              f"donated {ids.isin(accounts.loc[accounts['amount.donated.lifetime'] > 0, 'account.id']).mean():.1%}")

    print("\n=== Seasons covered ===")
    print(f"subscriptions: {subs['season'].min()} to {subs['season'].max()}")
    print(f"tickets: {tickets['season'].min()} to {tickets['season'].max()}")

    print("\n=== Share of missing values (account.csv) ===")
    print(accounts.isna().mean().round(3).to_string())


def plot_rate_by_recency(train, X_train, out_dir):
    """Subscription rate in 2014-15 vs. how long ago the patron last subscribed."""
    seasons_since = X_train["sub_seasons_since_last"]
    group = pd.cut(seasons_since, bins=[0, 1, 2, 3, 4, 100],
                   labels=["1 (2013-14)", "2", "3", "4", "5+"])
    group = group.cat.add_categories("never").fillna("never")
    rates = train.set_index("account.id")["label"].groupby(group, observed=False).agg(["mean", "size"])
    print("\n=== 2014-15 subscription rate by seasons since last subscription ===")
    print(rates.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rates.index.astype(str), rates["mean"] * 100, color=BLUE, width=0.6)
    ax.annotate(f"{rates['mean'].iloc[0]:.0%}", (0, rates["mean"].iloc[0] * 100),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                color="#0b0b0b")
    style_axes(ax)
    ax.set_xlabel("Seasons since last subscription", color=GRAY_TEXT)
    ax.set_ylabel("Subscribed in 2014-15 (%)", color=GRAY_TEXT)
    ax.set_title("Recent subscribers almost always renew", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "rate_by_recency.png", dpi=150)
    plt.close(fig)


def plot_history_coverage(train, test, accounts, subs, tickets, out_dir):
    """Do train and test have similar histories? (They should.)"""
    donors = accounts.loc[accounts["amount.donated.lifetime"] > 0, "account.id"]
    kinds = {"Has subscribed": subs["account.id"], "Has bought tickets": tickets["account.id"],
             "Has donated": donors}
    shares = pd.DataFrame({
        name: [ids.isin(kind_ids).mean() * 100 for kind_ids in kinds.values()]
        for name, ids in [("Train", train["account.id"]), ("Test", test["account.id"])]
    }, index=kinds.keys())

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(shares))
    ax.bar([i - 0.17 for i in x], shares["Train"], width=0.32, color=BLUE, label="Train")
    ax.bar([i + 0.17 for i in x], shares["Test"], width=0.32, color=ORANGE, label="Test")
    ax.set_xticks(list(x), shares.index)
    style_axes(ax)
    ax.set_ylabel("Share of accounts (%)", color=GRAY_TEXT)
    ax.set_title("Train and test have similar patron histories", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "history_coverage.png", dpi=150)
    plt.close(fig)


def main():
    train, test = load_data.load_train(), load_data.load_test()
    accounts = load_data.load_accounts()
    subs, tickets = load_data.load_subscriptions(), load_data.load_tickets()

    print_overview(train, test, accounts, subs, tickets)

    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train = build_features(train["account.id"])
    plot_rate_by_recency(train, X_train, out_dir)
    plot_history_coverage(train, test, accounts, subs, tickets, out_dir)
    print(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
