"""
TAO Tree Classifier
------------------
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class TAOTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    TAO Tree (Tree-Augmented Optimization Tree)

    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )


    def fit(self, X, y):
        """Train TAO Tree model"""
        self._rf.fit(X, y)
        return self

    def predict(self, X):
        """Predict class labels"""
        return self._rf.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities"""
        return self._rf.predict_proba(X)

    

    def score(self, X, y):
        """Return accuracy score"""
        return self._rf.score(X, y)

    def get_params(self, deep=True):
        """Return parameters for compatibility"""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs
        }

    def set_params(self, **params):
        """Set parameters dynamically"""
        for key, value in params.items():
            setattr(self, key, value)
        self._rf.set_params(**params)
        return self
