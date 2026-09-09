"""A reproducible three-class TF-IDF baseline; no pretrained downloads."""
import hashlib
import io
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS = ['negative', 'neutral', 'positive']

def normalize(value):
    # Preserve negation; punctuation-only variants should not cross the split.
    value = str(value).lower().replace('’', "'")
    value = re.sub(r"n\'t\b", ' not', value)
    value = re.sub(r'https?://\S+|www\.\S+|@\w+', ' ', value)
    value = re.sub(r'&amp;', ' and ', value)
    return ' '.join(re.findall(r'\b\w+\b', value))

def read_data(raw):
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError('Please provide a readable UTF-8 CSV.') from exc
    if {'text', 'airline_sentiment'}.issubset(df.columns):
        df = df.rename(columns={'text': 'review', 'airline_sentiment': 'sentiment'})
    if not {'review', 'sentiment'}.issubset(df.columns):
        raise ValueError('CSV needs review and sentiment columns.')
    df = df[['review', 'sentiment']].copy()
    if df.isna().any().any():
        raise ValueError('Remove rows with missing reviews or labels.')
    df['review'] = df.review.astype(str).str.strip()
    df['sentiment'] = df.sentiment.astype(str).str.strip().str.lower()
    if not set(df.sentiment).issubset(LABELS):
        raise ValueError('Labels must be positive, negative or neutral.')
    df['_key'] = df.review.map(normalize)
    original = len(df)
    empty = df['_key'].str.len() == 0
    empty_removed = int(empty.sum())
    df = df.loc[~empty].copy()
    conflicts = df.groupby('_key').sentiment.nunique()
    conflict_keys = conflicts[conflicts > 1].index
    conflict_removed = int(df['_key'].isin(conflict_keys).sum())
    df = df.loc[~df['_key'].isin(conflict_keys)].copy()
    before_dedup = len(df)
    df = df.drop_duplicates('_key').reset_index(drop=True)
    counts = df.sentiment.value_counts().reindex(LABELS, fill_value=0)
    if counts.min() < 10:
        raise ValueError('Provide at least 10 unique reviews for EACH of the three labels.')
    df.attrs['cleaning'] = {'input_rows': original, 'empty_removed': empty_removed,
                            'conflicting_rows_removed': conflict_removed,
                            'duplicates_removed': before_dedup - len(df)}
    return df, original - len(df)

def train(raw):
    df, removed = read_data(raw)
    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df.sentiment, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
        ('classifier', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42))
    ])
    pipeline.fit(train_df.review, train_df.sentiment)
    predicted = pipeline.predict(test_df.review)
    baseline = DummyClassifier(strategy='most_frequent').fit(train_df.review.to_frame(), train_df.sentiment)
    base_pred = baseline.predict(test_df.review.to_frame())
    report = {
        'dataset_sha256': hashlib.sha256(raw).hexdigest(),
        'seed': 42, 'train_rows': len(train_df), 'test_rows': len(test_df),
        'rows_removed': removed, 'duplicates_removed': df.attrs['cleaning']['duplicates_removed'],
        'cleaning': df.attrs['cleaning'], 'class_counts': df.sentiment.value_counts().to_dict(),
        'accuracy': accuracy_score(test_df.sentiment, predicted),
        'macro_f1': f1_score(test_df.sentiment, predicted, average='macro'),
        'baseline_accuracy': accuracy_score(test_df.sentiment, base_pred),
        'baseline_macro_f1': f1_score(test_df.sentiment, base_pred, average='macro'),
        'classification_report': classification_report(test_df.sentiment, predicted, labels=LABELS, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(test_df.sentiment, predicted, labels=LABELS).tolist(),
        'label_order': LABELS,
        'note': 'Single stratified row-level held-out split; related authors and paraphrases may cross splits. Results measure this dataset, not future or general product-review accuracy.'
    }
    test_predictions = test_df[['review', 'sentiment']].copy()
    test_predictions['prediction'] = predicted
    return pipeline, report, test_predictions, train_df['_key'].tolist(), test_df['_key'].tolist()

def predict(pipeline, review):
    if not isinstance(review, str) or not normalize(review):
        raise ValueError('Enter a review containing words.')
    if len(review) > 5000:
        raise ValueError('Keep reviews under 5,000 characters.')
    vector = pipeline.named_steps['tfidf'].transform([review])
    if vector.nnz == 0:
        raise ValueError('No familiar words found. Try an English airline or customer-service review.')
    probabilities = pipeline.predict_proba([review])[0]
    scores = dict(zip(pipeline.classes_, map(float, probabilities)))
    return max(scores, key=scores.get), scores
