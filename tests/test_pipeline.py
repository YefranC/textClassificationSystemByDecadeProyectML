from __future__ import annotations

import sys
from pathlib import Path
import unittest

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from text_classification.data import generate_synthetic_dataset, load_dataset
from text_classification.evaluation import evaluate_models
from text_classification.features import fit_transform_features
from text_classification.models import get_classifiers
from text_classification.preprocessing import preprocess_corpus, preprocess_text


class TextClassificationPipelineTests(unittest.TestCase):
    def test_generate_dataset_has_expected_columns(self):
        data = generate_synthetic_dataset(samples_per_decade=4)
        self.assertIn("text", data.columns)
        self.assertIn("label", data.columns)
        self.assertEqual(len(data), 16)

    def test_preprocess_text(self):
        self.assertEqual(preprocess_text("Hello,   WORLD!!!"), "hello world")

    def test_feature_extraction_shapes(self):
        texts = ["old school tunes", "viral social media"]
        vectorizer, x_train, x_test = fit_transform_features(texts, texts, ngram_range=(1, 2))
        self.assertGreater(x_train.shape[1], 0)
        self.assertEqual(x_train.shape[1], x_test.shape[1])
        self.assertIsNotNone(vectorizer)

    def test_evaluate_models_returns_scores(self):
        data = load_dataset(samples_per_decade=6)
        texts = preprocess_corpus(data["text"].tolist())
        labels = data["label"].tolist()
        x_train_texts, x_test_texts, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels,
        )
        _, x_train, x_test = fit_transform_features(x_train_texts, x_test_texts, ngram_range=(1, 2))
        results = evaluate_models(get_classifiers(), x_train, y_train, x_test, y_test)
        self.assertFalse(results.empty)
        self.assertEqual(set(results.columns), {"model", "accuracy", "f1_macro"})


if __name__ == "__main__":
    unittest.main()

