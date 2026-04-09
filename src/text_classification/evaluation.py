from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score


def evaluate_models(
    classifiers: Mapping[str, ClassifierMixin],
    x_train: spmatrix,
    y_train: Sequence[str],
    x_test: spmatrix,
    y_test: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name, model in classifiers.items():
        trained_model = clone(model)
        trained_model.fit(x_train, y_train)
        predictions = trained_model.predict(x_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "f1_macro": f1_score(y_test, predictions, average="macro"),
            }
        )
    results = pd.DataFrame(rows)
    results = results.sort_values(by=["f1_macro", "accuracy"], ascending=False)
    results = results.reset_index(drop=True)
    return results
