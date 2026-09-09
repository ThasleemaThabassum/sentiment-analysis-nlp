from pathlib import Path
import json
import pandas as pd
import streamlit as st
from model import train, predict, LABELS

st.set_page_config(page_title='ReviewLens | Sentiment analysis', page_icon='💬', layout='wide')

@st.cache_resource(show_spinner='Training TF–IDF + logistic regression…', max_entries=4)
def load_model(raw):
    return train(raw)

st.title('💬 ReviewLens')
st.markdown('### How was your airline experience?')
st.write('Explore positive, negative and neutral sentiment with a real NLP classifier.')
with st.sidebar:
    st.header('Training data')
    upload = st.file_uploader('Use your own labelled reviews', type=['csv'])
    st.caption('Upload Tweets.csv directly, or use review/sentiment columns. At least 10 unique reviews per class.')
    st.write('Model: TF–IDF + logistic regression')
    st.write('Features: words and two-word phrases')
    st.write('Split: 75% training / 25% testing')
    st.caption('English text baseline. Neutral means no clear positive or negative opinion—not uncertainty.')
raw = upload.getvalue() if upload else (Path(__file__).parent / 'data/reviews.csv').read_bytes()
try:
    pipeline, report, results, _, _ = load_model(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if not upload:
    st.info('Trained on your Kaggle airline dataset. Predictions are intended for English airline feedback; results may not transfer to other topics.')
else:
    st.caption('Using your uploaded CSV for this session.')
analyze, evaluate, learn = st.tabs(['Analyse a review', 'Model evaluation', 'How it works'])
with analyze:
    with st.form('review_form'):
        review = st.text_area('Enter a new review', placeholder='The crew was helpful and my flight arrived on time.', max_chars=5000, height=140)
        submitted = st.form_submit_button('Analyse sentiment', type='primary')
    if submitted:
        try:
            label, scores = predict(pipeline, review)
            st.subheader(f'Predicted sentiment: {label.capitalize()}')
            if max(scores.values()) < 0.55:
                st.warning('Low model confidence. Read this prediction cautiously; it is not automatically neutral.')
            chart = pd.DataFrame({'Model probability': [scores[x] for x in LABELS]}, index=LABELS)
            st.bar_chart(chart, horizontal=True)
            st.caption('Probabilities are model estimates, not calibrated guarantees of correctness.')
        except ValueError as exc:
            st.warning(str(exc))
    st.caption('Try: “Thank you for the excellent service!” / “My flight was cancelled and nobody helped.” / “What time does boarding start?”')
with evaluate:
    a, b, c = st.columns(3)
    a.metric('Test accuracy', f"{report['accuracy']:.1%}")
    b.metric('Macro-F1', f"{report['macro_f1']:.3f}")
    c.metric('Majority baseline accuracy', f"{report['baseline_accuracy']:.1%}")
    st.write(f"{report['train_rows']} training reviews · {report['test_rows']} held-out reviews · {report['rows_removed']} rows removed during cleaning")
    st.caption(report['note'])
    st.write('Cleaning summary', report['cleaning'])
    st.subheader('Confusion matrix')
    st.caption('Rows = actual labels. Columns = predicted labels.')
    st.dataframe(pd.DataFrame(report['confusion_matrix'], index=LABELS, columns=LABELS))
    st.subheader('Per-class performance')
    st.dataframe(pd.DataFrame(report['classification_report']).T.loc[LABELS])
    st.subheader('Held-out predictions')
    st.dataframe(results.reset_index(drop=True), hide_index=True)
    st.download_button('Download evaluation report', json.dumps(report, indent=2), 'metrics.json', 'application/json')
with learn:
    st.markdown('''1. **Label:** assign positive, negative or neutral to each review.
2. **Split:** remove empty text, conflicting labels and duplicates before separating training and test reviews.
3. **Represent:** fit TF–IDF on training reviews only, keeping negation and bigrams.
4. **Learn:** logistic regression learns a weight for each feature and class.
5. **Evaluate:** compare predictions with held-out labels and a majority baseline.
6. **Predict:** reuse the fitted pipeline for the text you enter.''')
    st.write('Limitations: sarcasm, mixed opinions, unfamiliar topics and non-English text. A single overall label can hide different opinions about different features.')
    st.caption('The model is trained in memory at launch and cached. Uploaded datasets are not written to disk by this app.')
