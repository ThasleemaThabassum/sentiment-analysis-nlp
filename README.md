ReviewLens | Airline Sentiment Analysis

An NLP application that classifies English airline feedback as positive, negative, or neutral using TF–IDF and logistic regression, with an interactive Streamlit interface.

Launch the live app · Explore the dataset

Overview

Customer feedback contains useful signals about service quality, delays, cancellations, and travel experiences. ReviewLens demonstrates how a supervised text-classification pipeline can turn that feedback into sentiment labels.

The project covers data preparation, feature extraction, model training, held-out evaluation, and a deployed interface. It is designed as an educational baseline for airline feedback analysis.

Features

Review prediction: enter feedback and receive a three-class sentiment prediction.

Probability chart: inspect the model’s estimated probability for each class.

Low-confidence indication: distinguish uncertainty from neutral sentiment.

Custom dataset upload: train on a CSV using either review / sentiment or Kaggle’s text / airline_sentiment columns.

Evaluation dashboard: explore accuracy, macro-F1, per-class metrics, and a confusion matrix.

Error inspection: view individual held-out predictions and download an evaluation report.

Technology Stack

Component

Technology

Language

Python

Data processing

pandas

Text features

scikit-learn TF–IDF

Classifier

scikit-learn logistic regression

Interface and charts

Streamlit

Hosting

Streamlit Community Cloud

Dataset

The project uses Twitter US Airline Sentiment, shared by CrowdFlower on Kaggle. It contains 14,640 tweets labelled positive, negative, or neutral and covers multiple US airlines—not only American Airlines.

Dataset source and usage terms

Original label

Tweets

Negative

9,178

Neutral

3,099

Positive

2,363

Total

14,640

Only tweet text is used as model input. The sentiment label is the target. Other fields, including annotation confidence and negative-reason labels, are excluded from the features.

Preparation

Text normalization removes URLs and user mentions, normalizes case, and retains negation. Before splitting, the pipeline removes all rows in conflicting normalized-text label groups and then removes remaining duplicates.

Preparation result

Rows

Original dataset

14,640

Conflicting-label rows removed

261

Additional duplicate rows removed

240

Empty normalized rows removed

0

Available for modelling

14,139

Model Workflow

Prepare the data: validate labels and clean duplicate or conflicting examples.

Split the dataset: use a stratified 75% training / 25% test split with random seed 42.

Extract features: fit TF–IDF on training text only, using unigrams and bigrams with a maximum of 20,000 features.

Train the classifier: fit logistic regression with balanced class weights.

Evaluate: compare held-out predictions with the labelled test set and a majority-class baseline.

Predict: reuse the fitted pipeline for new feedback entered in the app.

The app trains the model in memory and caches it. No external AI API or pretrained model download is required.

Evaluation Results

The following results were measured on a single held-out test split using the bundled dataset.

Metric

Result

Training samples

10,604

Test samples

3,535

Test accuracy

78.42%

Macro-F1

0.729

Majority-class baseline accuracy

64.07%

Performance by Class

Sentiment

Precision

Recall

F1-score

Negative

0.888

0.837

0.862

Neutral

0.584

0.669

0.624

Positive

0.688

0.717

0.702

Negative sentiment is the strongest class in this evaluation. Neutral sentiment remains more difficult. Macro-F1 is included because it gives equal weight to each class, while accuracy can be dominated by the larger negative class.

These results describe this dataset and split. They are not a guarantee of performance on future feedback or other domains.

Run Locally

Use Python 3.12 to match the original development setup. Download and extract the repository, then open a terminal in the folder containing app.py and requirements.txt.

Windows PowerShell

py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py

macOS / Linux

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py

Open http://localhost:8501 if the browser does not open automatically. The first launch trains the model; subsequent interactions reuse the cached model.

Dataset location: this repository’s deployed app.py expects reviews.csv beside app.py. Keep that layout when downloading and running the GitHub version.

Use a Custom Dataset

Upload a UTF-8 CSV from the app’s sidebar. Supported column pairs are:

Format

Text column

Label column

General CSV

review

sentiment

Kaggle airline CSV

text

airline_sentiment

Accepted labels are positive, negative, and neutral. At least 10 unique, usable examples per class are required after cleaning; a much larger, independently labelled dataset is preferable for meaningful evaluation.

Format example only:

review,sentiment
"The crew was helpful and friendly.",positive
"My flight was cancelled without support.",negative
"What time does boarding begin?",neutral

The three rows above illustrate the format and are not enough to train the model.

Reproduce the Report

For the repository layout with reviews.csv at the root, run:

.venv\Scripts\python train.py --data reviews.csv

The explicit --data argument overrides the training script’s original default dataset path. Evaluation outputs are written to reports/metrics.json and reports/test_predictions.csv.

Limitations

The training data concerns US airline tweets; results may not transfer to other products, languages, or writing styles.

Sarcasm, mixed opinions, and unfamiliar expressions can cause errors.

One overall label may hide different opinions about separate aspects of a journey.

Model probabilities are uncalibrated estimates, not guarantees. Low confidence does not mean neutral sentiment.

Duplicate removal reduces exact-text leakage, but related authors and paraphrases may still occur across the row-level split.

Stronger evaluation would use independent data or time-based/author-grouped splits. Any tuning should use training-only validation, leaving the final test set untouched.

Possible Extensions

Aspect-level analysis for staff, baggage, delays, and booking experience.

Better coverage of neutral and mixed feedback.

Comparison with other classical baselines and contextual language models.

Probability calibration and evaluation on more recent, independently collected data.

These are future directions, not implemented features.

Author

Thasleema Thabassum
B.Tech — Computer Science and Engineering (Data Science)

References

Twitter US Airline Sentiment — Kaggle / CrowdFlower

TF–IDF vectorizer — scikit-learn

Logistic regression — scikit-learn

Model evaluation — scikit-learn

Streamlit documentation
