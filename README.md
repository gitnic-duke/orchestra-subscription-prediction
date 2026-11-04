# Orchestra Season Subscription Prediction

Class Kaggle competition (AIPI 520, Duke): predict which past patrons of an
orchestra will buy a season subscription for the 2014-15 season. Predictions
are probabilities, and the score is the area under the ROC curve (AUROC).

> **Write-up:** the full description of the modeling process (data pipeline,
> cleaning, features, algorithms, validation, hyperparameters, and the reasoning
> behind each decision) is in [`writeup.md`](writeup.md). This README covers
> how to run the code.

## Final model

- **Features:** 8 features built from subscription history, single-ticket
  purchases, donation history and distance to the concert venues
  (`SELECTED_FEATURES` in `src/features.py`).
- **Model:** a 50/50 average of the predicted probabilities of two tuned models,
  gradient boosting (scikit-learn `HistGradientBoostingClassifier`) and logistic
  regression (`src/models.py`).
- **Cross-validated AUROC:** 0.9812 (5-fold stratified CV, repeated 3 times).

## Repository layout

```
.
├── data/                     # raw Kaggle CSVs (not committed, see "Setup")
├── src/
│   ├── config.py             # paths and constants (random seed, target season)
│   ├── load_data.py          # reading and cleaning each raw file
│   ├── features.py           # builds one feature row per account (train and test)
│   ├── models.py             # model definitions, including the ensemble
│   ├── evaluate.py           # cross-validation and the experiment log
│   └── submit.py             # writes and checks Kaggle submission files
├── eda.py                    # exploratory analysis, saves figures
├── baseline.py               # rule-based baseline (no ML)
├── experiments.py            # named experiments: features + model -> CV AUROC
├── feature_selection.py      # feature-group ablation and permutation importance
├── tune.py                   # hyperparameter search
├── ensemble.py               # tests weighted averages of the two tuned models
├── make_submission.py        # trains on all training data, writes a submission
├── results/
│   ├── cv_results.csv        # log of every experiment's CV AUROC
│   ├── feature_importance.csv
│   ├── tuning_*.csv          # every hyperparameter candidate's CV AUROC
│   └── figures/              # charts from eda.py and feature_selection.py
├── submissions/              # prediction files uploaded to Kaggle
└── writeup.md                # write-up of the modeling process
```

## Setup

1. Python 3.10+ and the packages in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the competition data from Kaggle and put the CSV files in `data/`:
   `train.csv`, `test.csv`, `sample_submission.csv`, `account.csv`,
   `subscriptions.csv`, `tickets_all.csv`, `concerts.csv`,
   `concerts_2014-15.csv`, `zipcodes.csv`.

## Reproducing the final submissions

Run from the repository root:

```bash
python make_submission.py ensemble_prob_50_50   # final model (ensemble)
python make_submission.py hgb_tuned_selected    # tuned gradient boosting alone
```

Each command builds the features for train and test, trains the model on all
6,941 training accounts, and writes `submissions/<name>.csv` in the format of
`sample_submission.csv`. All randomness uses a fixed seed (42), so the files
are reproduced exactly.

## Reproducing the analysis

The scripts below were run in this order. Each one prints its results and
appends CV scores to `results/cv_results.csv`.

```bash
python eda.py                     # data summary and figures
python baseline.py                # rule-based baseline
python experiments.py --list      # list all named experiments
python experiments.py <name>      # run one experiment
python experiments.py <a> <b>     # run two and compare them fold by fold
python feature_selection.py       # group ablation, permutation importance (~1 min)
python tune.py                    # hyperparameter search (~3 min)
python ensemble.py                # ensemble weights and averaging methods (~1 min)
```

## Validation

Every experiment uses the same 5-fold stratified cross-validation, repeated 3
times with a fixed seed (`src/evaluate.py`), giving 15 AUROC scores per
experiment. Because the folds are identical, two experiments can be compared
fold by fold. Model choices were made on CV AUROC, not on the public
leaderboard.

## Results

| Step | Approach | Features | CV AUROC | Public LB |
|---|---|---|---|---|
| Baseline | Rule: rank by most recent season subscribed | 2 | 0.9351 | |
| 1 | Gradient boosting, default settings | 4 | 0.9512 | |
| 1 | Gradient boosting, regularized | 15 | 0.9585 | |
| 2 | + single-ticket features | 23 | 0.9652 | |
| 3 | + donation and account features | 30 | 0.9769 | |
| 4–5 | + distance to venues | 31 | 0.9781 | 0.97113 |
| 5 | Logistic regression / random forest (same 31 features) | 31 | 0.9748 / 0.9716 | |
| 6 | Feature selection | 8 | 0.9797 | |
| 7 | Tuned gradient boosting | 8 | 0.9805 | 0.97157 |
| 7 | Tuned logistic regression | 8 | 0.9785 | |
| 8 | **Ensemble: 50/50 average of tuned boosting + logistic regression** | 8 | **0.9812** | 0.97125 |

The full log of every experiment is in `results/cv_results.csv`.
