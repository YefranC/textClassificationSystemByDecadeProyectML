from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def get_classifiers(random_state: int = 42):
    return {
        "naive_bayes": MultinomialNB(),
        "svm": LinearSVC(random_state=random_state),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
    }

