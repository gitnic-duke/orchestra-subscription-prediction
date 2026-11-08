# Predicting 2014-2015 Season Subscriptions: Writeup

## 1. Summary
For my final submission on Kaggle, I ensembled the results of a gradient boosting model and a logistic regression model, using 8 features based on subscription history, ticket purchases, donations, and distance to concert venues. It scored 0.9812 AUROC in local cross-validation and 0.97125 on the public Kaggle set.

## 2. Initial Data Exploration
Initial data exploration showed 6,941 accounts and only 332 subscribers among them, meaning we had a large class imbalance, indicating AUROC was an appropriate scoring method rather than basic accuracy. For many of the customers there wasn't much helpful information since only about a third had any subscription history, a tenth bought single tickets, and a quarter had ever donated. The most important piece of data for predicting future subscriptions was if/how recently the customer last subscribed. 82% of 2013-14 subscribers renewed, while 13%, 8%, and 5% renewed if they subscribed 2, 3, or 4 seasons earlier. If it was 5 or more seasons ago they didn't come back. 

This indicated to me that how recently the customer last subscribed would be an important feature, and I used a heuristic baseline based on most recent subscription season that achieved 0.935 cross-validation AUROC to compare future iterations of my model to.

However, customers who never subscribed still bought subscriptions at 0.7%, so there would be room for improvement for the model to predict which of those customers would buy a subscription.

## 3. Validation Approach
I used 5-fold stratified cross-validation repeated 3 times, using a fixed seed to keep the folds consistent across different approaches. Reasoning and approach:
- Stratifying was necessary to keep the ~5% positive rate for subscribing customers. A random split could result in a fold with too few subscribers
- I chose to repeat 3 times because each validation fold would only have about 66 subscribers, so averaging scores could smoothen out any noise caused by small sample size.
- For each different approach I would test, I kept the folds the same so I could compare fold by fold performance. That's opposed to doing something like comparing means, in order to keep more sensitivity since some folds could be more difficult model to model. Between different approaches, I also counted how many of the 15 folds improved (differences under ~0.003 were treated as noise) to help me make a decision on which one to choose.
- Whenever I picked the best of several options, I re-tested the winner on three fresh sets of folds as a sanity check.
- I decided to use the cross-validation results to choose my model instead of the public leaderboard since the leaderboard is roughly the size of one CV fold, which means it's about as noisy as one fold.

## 4. Data Pipeline and CLeaning
`load_data.py` reads and cleans each file, `features.py` builds one row per customer account, `models.py` defines the models, `evaluate.py` runs cross-validation, and `make_submission.py` retrains on all the training data and outputs predictions for the test set, in the format necessary for Kaggle submission.

During cleaning the main steps were:
- converting season labels like "2013-2014" to be a start year instead
- reading the csv as latin-1 encoding since I ran into issues reading it as UTF-8
- ensuring zip codes would be five digits (restoring dropped leading zeros)
- using the billing zip code since the shipping zip code was missing for almost all of the accounts
- set text labels in the ticket price-level column as missing. This was a messy column that included some useful numeric information indicating the price level, and it was easier to just remove the text labels so the model could use mean ticket price level without worrying about text values
- grouping rare category derivatives into one category (e.g. "Full upgrade" -> "Full") because they indicate the same or similar types of packages bought that could be useful for the model to group together

Missing values were left missing on purpose since it could be informative to see it as missing (e.g. no subscription history). Missing counts were filled with 0 since that's what the data seemed to indicate missing meant.

## 5. Features
I tested features in one categorical group at a time, keeping each only if the CV AUROC improved. These categories seemed to make intuitive sense that differences in them could affect subscription purchase rate.

Feature groups:
- Subscription history (15 features): Subscription recency, tenure, streak length, details of latest subscription. Recency was the most useful. Details such as seats, tier, and package didn't seem to add any measureable improvement to AUROC.
- Single tickets (8 features): Ticket mean price level, number of ticket purchases, seasons since last ticket purchase, ticket bought in 2013, family concert tickets bought, number of seasons with a ticket purchase, total seats bought, most recent season with a bought ticket. Adding this group added 0.007 AUROC and improved 12 of 15 folds during CV. Reading into the results, ticket buyers who weren't recent subscribers subscribed more often than those that didn't buy tickets, and buyers of pricier seats also subscribed more.
- Donations and account info (7 features): Year of first donation, number of lifetime donations, includes account billing zip, account relationship, donation amount in 2013, average donation amount, lifetime donation amount. In total these added 0.012 AUROC and improved all 15 folds. In the final, only first year of donation and number of lifetime donations were kept. 
- Geography (4 features): Miles to nearest of concert venues, average wage in zip, population in zip, if the zip was in California. Distance to nearest venue was the only one that helped.

A caveat with the donation features: it seemed like there was potential data leakage in the data given, since some first donations were labelled as being in 2014. These potentially looked like donations added when buying the subscription, which is definitely data leakage, but since they were a part of the data given I decided to keep it in.

Feature selection:

I tried removing each group one at a time, and that showed all of them contributed (subscriptions - 0.058, donations -0.004, tickets -0.002, distance -0.001). I then tried shuffling one feature to see how much validation AUROC drops which showed 15 of 31 features scored 0 or below, so I removed them. Looking at the data I had, I noticed a jump in importance between the 8th and 9th ranked features (0.00031 and 0.00016 improvement in AUROC), and I decided to keep just the top 8 features for simplicity and to lower overfitting risk.

Final features:

| Feature | Meaning |
|---|---|
| `sub_last_season` | most recent season subscribed (by far the most important) |
| `sub_first_season` | first season subscribed |
| `sub_last_streak` | length of the unbroken run ending at the last subscription |
| `sub_last_location` | venue of the last subscription |
| `tix_mean_price_level` | average price level of single tickets bought |
| `don_first_year` | year of first donation |
| `don_n_lifetime` | number of donations ever made |
| `geo_miles_to_concerts` | distance to the nearest concert city; missing if no address |


## 6. Models
I started with scikit-learn's `HistGradientBoostingClassifier` as the main model since it handles missing values and categorical columns on its own. As a boosted tree type model, it can also capture interactions without any manual processing from me. I then compared performance on the original 31 features and the same folds with logistic regression (0.9748 AUROC) and random forest (0.9716). Logistic regression was chosen to see how a simple, interpretable, linear type model could perform. Random forest was chosen to see if a flexible non-linear model could help, in the bagged trees family of models. Gradient boosting (0.9781) ended up being the winner. During the testing, logistic regression needed preprocessing in order to achieve that AUROC: log-scaling dollar amounts, median imputation with "missing" indicator columns, standardization, and one-hot encoding.

Some models I avoided:
- KNN - this relies on distances which works badly with missing values, mixed categories, and skewed scales.
- SVM - doesn't directly give probabilities, needs similar preprocessing as logistic regression (extra work), and rarely beats boosting on small tabular data like we were given
- Naive Bayes - assumes features are independent, but ours are correlated
- Neural network - needs much more data

Regularization and tuning:

Early on, the parameters seemed to favor shallower trees and a slower learning rate. So I did a randomized search over 60 combinations of six settings for gradient boosting, focusing the combinations on the shallower trees and slower learning rate. For logistic regression, I focused testing on different values over regularization strength C.

Final parameters:

| Model | Final hyperparameters |
|---|---|
| Gradient boosting | max_depth 2, learning_rate 0.05, max_iter 400, min_samples_leaf 20, l2_regularization 10, max_features 0.75 |
| Logistic regression | C = 3, L2 penalty |
| Ensemble | 50/50 average of the two models' predicted probabilities |

For ensemble, I noticed the two tuned models had similar scores, but they work differently so I wanted to keep both. I tested weighted averages of their ranks, which I expected to work best since AUROC only depends on order. However, averaging probabilities matched or beat averaging ranks at every weight, with 50/50 being best (also tested 30/70 and 70/30). It also beat boosting alone. The reason for this is probably the class imbalance. Almost 90% of the customers got a boosting probability under 1% and almost none of them subscribe, but ranking spreads them evenly, so meaningless differences among near certain non-subscribers count as much as a big gap (0.3% and 30% could be treated similarly in ranking). Averaging probabilities instead of focusing on ranks keeps that real gap in there.

For feature selection, I retested a couple of feature sets to make sure tuning didn't change anything. Notably, adding all 31 features as opposed to the 8 features decreased performance more than before, so I kept the original 8 features since they had the best performance.

## 7. Results
| Step | Approach | Features | CV AUROC | Public Submission |
|---|---|---|---|---|
| Baseline | Rank by most recent season subscribed | 2 | 0.9351 | |
| Subscriptions | Gradient boosting, regularized | 15 | 0.9585 | |
| + Tickets | | 23 | 0.9652 | |
| + Donations | Data as given (leak-free version: 0.9722) | 30 | 0.9769 | |
| + Distance | | 31 | 0.9781 | 0.97113 |
| Feature selection | | 8 | 0.9797 | |
| Tuning | Tuned gradient boosting | 8 | 0.9805 | 0.97157 |
| **Final** | **50/50 ensemble with tuned logistic regression** | **8** | **0.9812** | **0.97125** |

Even though among my public submissions, tuned gradient boosting won, it was only by a very slim margin, so I went with the 50/50 ensemble which scored higher on my local CV AUROC testing.

Final notes:
- I could've experimented more with the logistic regression tuning, but due to time constraints and the fact my score was already high enough, I decided not to. But tuning logistic regression may have slightly improved AUROC.
- I didn't try all the data availble (concert programs). Matching customer's past concerts to the 2014-2015 program might've been useful.
- Leakage: as mentioned, the model uses donation data that may have come in the future. Without that, we may have gotten a lower AUROC
- Due to the class imbalance and small number of positives, a lot of the differences in the results of my experiments may have been due to noise. Where that happened, I mostly decided on simplicity and whether that result stayed true across many folds.