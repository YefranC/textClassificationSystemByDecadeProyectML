from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from text_classification.data import load_dataset
from text_classification.evaluation import evaluate_models
from text_classification.features import fit_transform_features
from text_classification.models import get_classifiers
from text_classification.preprocessing import preprocess_corpus


def run(dataset_path: str | None = None, test_size: float = 0.2):
    data = load_dataset(dataset_path)
    texts = preprocess_corpus(data["text"].astype(str).tolist())
    labels = data["label"].astype(str).tolist()

    x_train_texts, x_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    _, x_train, x_test = fit_transform_features(x_train_texts, x_test_texts, ngram_range=(1, 2))
    results = evaluate_models(get_classifiers(), x_train, y_train, x_test, y_test)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Text classification by decade")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to CSV with text,label columns")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    results = run(dataset_path=args.dataset, test_size=args.test_size)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

