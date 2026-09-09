# ReviewLens — Airline Sentiment Analysis Using NLP

Updated with the Twitter US Airline Sentiment dataset uploaded by you.
A real TF–IDF + logistic regression classifier predicts positive, negative or neutral
sentiment in English airline feedback. Includes Streamlit, CSV upload, evaluation,
and reproducible training. No API key required.

## Start on Windows

Extract the ZIP. Open the `sentiment_nlp` folder in VS Code, then open its terminal.
Install Python 3.11 or 3.12 if needed, and run:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically. The model trains
at launch and is cached. Keep the terminal open; Ctrl+C stops the app. Internet is
needed to install packages; the included data then works offline.

macOS/Linux: use `python3 -m venv .venv`, then `.venv/bin/python` in place of
`.venv\Scripts\python` in the commands above.

## What changed

- Replaced 120 synthetic examples with your real, 14,640-row Kaggle CSV.
- Directly accepts Kaggle `text`/`airline_sentiment` columns or `review`/`sentiment`.
- Removes mentions and URLs for modelling, retains negation and word bigrams.
- Removes all rows in conflicting normalized-text label groups, instead of guessing labels.
- Removes remaining normalized duplicates before the split.
- Updated the interface for airline feedback and included actual evaluation reports.

## Actual evaluation

| Measure | Result |
|---|---:|
| Original rows | 14,640 |
| Conflicting-label rows removed | 261 |
| Additional duplicate rows removed | 240 |
| Empty normalized rows removed | 0 |
| Remaining rows | 14,139 |
| Training rows | 10,604 |
| Held-out test rows | 3,535 |
| Test accuracy | 78.42% |
| Macro-F1 | 0.7293 |
| Majority-class baseline accuracy | 64.07% |

Per-class F1: negative 0.8616, neutral 0.6240, positive 0.7023.
Neutral is harder for this model than negative. Scores are from one fixed stratified
75/25 row split, seed 42; hyperparameters were not tuned against this test set.
TF–IDF vocabulary/IDF are fitted only on training text. Logistic regression uses
balanced class weights, max_iter=1500 and at most 20,000 unigram/bigram features.

These scores are not directly comparable with the earlier synthetic-data score because
the test datasets differ. They are not a guarantee for future reviews, other languages,
or product reviews. Related authors and paraphrases may still cross this row-level
split. For a stronger study, reserve an independent time-based or author-grouped test
set and use training-only cross-validation for tuning. Do not tune against the test tab.

## Using the app

Enter an airline review, click **Analyse sentiment**, and read its predicted label.
The bar chart displays uncalibrated class probabilities, not guaranteed correctness.
Low confidence is indicated separately from neutral sentiment. The evaluation tab
shows the confusion matrix, class metrics, individual held-out predictions and a report
download. The sidebar accepts another labelled CSV; at least 10 cleaned reviews per
class are required. Missing values and unsupported labels produce an actionable error.
Uploaded data is processed in memory; this app does not write it to disk.

Example inputs (original demo sentences):
- Thank you for the excellent service!
- My flight was cancelled and nobody helped.
- What time does boarding start?

Sarcasm, mixed opinions and unfamiliar topics remain limitations. Overall polarity is
not an aspect-level sentiment analysis system.

## Reproduce the evaluation

```powershell
.venv\Scripts\python train.py
.venv\Scripts\python -m unittest test_project -v
```

`train.py --data path/to/Tweets.csv` accepts a different dataset. Reports are written to
`reports/metrics.json` and `reports/test_predictions.csv`. The model is rebuilt in memory;
no pickle or pretrained model download is needed. Exact tested main-package versions
are in `requirements-tested.txt`; `requirements.txt` contains compatible ranges.

## Files

- app.py — Streamlit UI.
- model.py — normalization, validation, deduplication, training and prediction.
- train.py — reproducible evaluation export.
- test_project.py — pipeline and Streamlit interaction tests.
- data/reviews.csv — your uploaded Tweets.csv, retaining original columns.
- data/SOURCE.txt — dataset provenance and transformation policy.
- reports/ — actual measured evaluation, not illustrative numbers.

Only text is passed to the vectorizer and only airline_sentiment supplies targets.
Other columns such as label confidence, negative reason, user name and location are
not used as model features. The SQLite copy from the upload is not needed.

## Explain it in your seminar

“I used the Kaggle Twitter US Airline Sentiment dataset. After cleaning duplicates and
conflicting labels, I trained TF–IDF with logistic regression on 10,604 tweets. On 3,535
held-out tweets, it achieved 78.42% accuracy and 0.729 macro-F1. The Streamlit interface
accepts new airline feedback and predicts positive, negative or neutral sentiment.”

## References

- Dataset: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- TF–IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Streamlit: https://docs.streamlit.io/
