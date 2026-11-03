# Orchestra Season Subscription Prediction

Class Kaggle competition (AIPI 520, Duke): predict which past patrons of an
orchestra will buy a season subscription for the 2014-15 season.
Predictions are probabilities, and the score is the area under the ROC
curve (AUROC).

The full write-up of the modeling process is in [`writeup.md`](writeup.md).

## Repository layout

```
.
├── data/                 # raw Kaggle CSVs (not committed, see "Setup")
├── src/
│   ├── config.py         # paths and constants (random seed, target season)
│   ├── load_data.py      # reading and cleaning each raw file
│   ├── features.py       # builds one feature row per account (train and test)
│   ├── evaluate.py       # cross-validation and the experiment log
│   └── submit.py         # writes Kaggle submission files
├── eda.py                # exploratory analysis, saves figures
├── baseline.py           # rule-based baseline (no ML)
├── results/
│   ├── cv_results.csv    # log of every experiment's CV AUROC
│   └── figures/          # charts from eda.py
└── submissions/          # prediction files uploaded to Kaggle
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

## Usage

Run every script from the repository root:

```bash
python eda.py        # print data summary, save figures to results/figures/
python baseline.py   # evaluate the baseline rule and write a submission
```

Each run appends its cross-validated AUROC to `results/cv_results.csv` and
writes its predictions to `submissions/`.

## Validation

All experiments use the same 5-fold stratified cross-validation, repeated
3 times with a fixed seed (`src/evaluate.py`), and are compared on mean
CV AUROC.

## Results

| Approach | CV AUROC |
|---|---|
| Baseline rule (recency of last subscription) | 0.935 |
