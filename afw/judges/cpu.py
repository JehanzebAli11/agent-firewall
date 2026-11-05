"""CPU judges: TF-IDF + logistic regression, small MLP, gradient boosting.
All output P(action does NOT follow from the user's request)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .features import make_text, struct_matrix


class _Base:
    name = "base"
    eval_only = False

    def __init__(self, view="full", seed=0):
        self.view, self.seed = view, seed

    def _text_fit(self, recs):
        self.wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000, sublinear_tf=True)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=120_000, sublinear_tf=True)
        texts = [make_text(r, self.view) for r in recs]
        return sp.hstack([self.wv.fit_transform(texts), self.cv.fit_transform(texts)]).tocsr()

    def _text(self, recs):
        texts = [make_text(r, self.view) for r in recs]
        return sp.hstack([self.wv.transform(texts), self.cv.transform(texts)]).tocsr()

    def logit(self, recs) -> np.ndarray:
        p = np.clip(self.predict_proba(recs), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))


class TfidfLR(_Base):
    name = "tfidf_lr"

    def fit(self, recs, y):
        X = self._text_fit(recs)
        self.sc = StandardScaler()
        S = sp.csr_matrix(self.sc.fit_transform(struct_matrix(recs, self.view)))
        self.m = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear")
        self.m.fit(sp.hstack([X, S]).tocsr(), y)
        return self

    def _X(self, recs):
        return sp.hstack([self._text(recs), sp.csr_matrix(self.sc.transform(struct_matrix(recs, self.view)))]).tocsr()

    def predict_proba(self, recs):
        return self.m.predict_proba(self._X(recs))[:, 1]

    def logit(self, recs):
        return self.m.decision_function(self._X(recs))


class _Dense(_Base):
    n_svd = 256

    def _dense_fit(self, recs):
        X = self._text_fit(recs)
        self.svd = TruncatedSVD(self.n_svd, random_state=self.seed)
        Z = self.svd.fit_transform(X)
        self.sc = StandardScaler()
        return self.sc.fit_transform(np.hstack([Z, struct_matrix(recs, self.view)]))

    def _dense(self, recs):
        Z = self.svd.transform(self._text(recs))
        return self.sc.transform(np.hstack([Z, struct_matrix(recs, self.view)]))

    def predict_proba(self, recs):
        return self.m.predict_proba(self._dense(recs))[:, 1]


class SmallMLP(_Dense):
    name = "mlp"

    def fit(self, recs, y):
        D = self._dense_fit(recs)
        self.m = MLPClassifier((256, 64), alpha=1e-3, batch_size=256, max_iter=60, early_stopping=True,
                               n_iter_no_change=5, random_state=self.seed)
        self.m.fit(D, y)
        return self


class GBM(_Dense):
    name = "gbm"
    n_svd = 128

    def fit(self, recs, y):
        D = self._dense_fit(recs)
        self.m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                                early_stopping=True, random_state=self.seed)
        self.m.fit(D, y)
        return self


JUDGES = {"tfidf_lr": TfidfLR, "mlp": SmallMLP, "gbm": GBM}
