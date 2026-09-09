import io
import unittest
from pathlib import Path
import pandas as pd
from model import train, predict, read_data, normalize

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (Path(__file__).parent / 'data/reviews.csv').read_bytes()
        cls.model, cls.report, _, cls.training, cls.testing = train(cls.raw)

    def fixture(self):
        return pd.DataFrame([{'review':f'{label} example number {i}', 'sentiment':label}
            for label in ['positive','negative','neutral'] for i in range(12)])

    def test_no_split_leakage(self):
        self.assertFalse(set(self.training) & set(self.testing))
        self.assertEqual(self.report['train_rows'] + self.report['test_rows'] + self.report['rows_removed'], 14640)
        self.assertEqual(self.report['cleaning']['conflicting_rows_removed'],261)

    def test_scores_are_three_class_probabilities(self):
        label, scores = predict(self.model, 'Thank you for the excellent service!')
        self.assertEqual(set(scores), {'positive','negative','neutral'})
        self.assertIn(label, scores)
        self.assertAlmostEqual(sum(scores.values()), 1)

    def test_invalid_reviews(self):
        for review in ['', '!!!', 'qzxvplm', 'x'*5001]:
            with self.assertRaises(ValueError): predict(self.model, review)

    def test_conflicts_removed_without_guessing(self):
        frame=self.fixture()
        frame=pd.concat([frame,pd.DataFrame([{'review':'positive example number 0','sentiment':'negative'}])])
        cleaned, removed=read_data(frame.to_csv(index=False).encode())
        self.assertEqual(removed,2)
        self.assertNotIn('positive example number 0',cleaned['_key'].tolist())

    def test_duplicates_after_handle_and_url_removal(self):
        frame=self.fixture()
        frame=pd.concat([frame,pd.DataFrame([{'review':'@airline positive example number 0 https://example.com','sentiment':'positive'}])])
        cleaned, removed=read_data(frame.to_csv(index=False).encode())
        self.assertEqual(removed,1)
        self.assertEqual(len(cleaned),36)

    def test_kaggle_schema(self):
        frame=self.fixture().rename(columns={'review':'text','sentiment':'airline_sentiment'})
        cleaned,_=read_data(frame.to_csv(index=False).encode())
        self.assertEqual(len(cleaned),36)

    def test_missing_class(self):
        with self.assertRaises(ValueError): read_data(b'review,sentiment\nhello,neutral\n')

    def test_training_only_vocabulary(self):
        vectorizer = self.model.named_steps['tfidf']
        analyzer = vectorizer.build_analyzer()
        training_features = set()
        for review in self.training:
            training_features.update(analyzer(review))
        self.assertTrue(set(vectorizer.vocabulary_).issubset(training_features))

    def test_streamlit_prediction_and_empty_input(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).parent/'app.py')).run(timeout=60)
        self.assertFalse(app.exception)
        app.text_area[0].set_value('My flight was cancelled and nobody helped.')
        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any('Predicted sentiment:' in x.value for x in app.subheader))
        app.text_area[0].set_value('')
        app.button[0].click().run(timeout=60)
        self.assertTrue(any('Enter a review' in x.value for x in app.warning))

if __name__ == '__main__': unittest.main(verbosity=2)
