from __future__ import annotations

from typing import Iterable

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int | None = 5000,
) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)


def fit_transform_features(
    train_texts: Iterable[str],
    test_texts: Iterable[str],
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int | None = 5000,
) -> tuple[TfidfVectorizer, spmatrix, spmatrix]:
    vectorizer = create_vectorizer(ngram_range=ngram_range, max_features=max_features)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return vectorizer, x_train, x_test
