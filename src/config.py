"""
Project-wide settings: file locations and constants shared by every script.
"""
from pathlib import Path

# Repo root = the folder that contains src/
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"                # raw Kaggle CSVs go here (not committed)
SUBMISSION_DIR = ROOT / "submissions"   # prediction files we upload to Kaggle
RESULTS_DIR = ROOT / "results"          # cross-validation log and figures

# Fixed seed so cross-validation folds and models are reproducible
RANDOM_SEED = 42

# We predict subscriptions for the 2014-15 season. Every feature must use only
# information from seasons that started before this year (2013-14 and earlier).
TARGET_SEASON_START = 2014
