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


class Featurizer:
    """TF-IDF (word 1-2 + char 2-4) + structured features; optional SVD for dense models.
    One instance can be shared by several judges trained on the same split (fit once)."""

    def __init__(self, view="full", seed=0, n_svd=192):
        self.view, self.seed, self.n_svd = view, seed, n_svd
        self.fitted, self.svd = False, None

    def fit(self, recs):
        if self.fitted:
            return self
        texts = [make_text(r, self.view) for r in recs]
        self.wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True, dtype=np.float32)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=60_000, sublinear_tf=True, dtype=np.float32)
        self.wv.fit(texts)
        self.cv.fit(texts)
        S = struct_matrix(recs, self.view)
        self.sc = StandardScaler().fit(S)
        self.fitted = True
        self._cache_key, self._cache = id(recs), None
        return self

    def sparse(self, recs):
        texts = [make_text(r, self.view) for r in recs]
        return sp.hstack([self.wv.transform(texts), self.cv.transform(texts)]).tocsr()

    def struct(self, recs):
        return self.sc.transform(struct_matrix(recs, self.view))

    def dense(self, recs, fit=False):
        # SVD over WORD tf-idf only: the char n-gram block is ~10x denser and makes SVD the bottleneck
        X = self.wv.transform([make_text(r, self.view) for r in recs])
        if self.svd is None:
            self.svd = TruncatedSVD(self.n_svd, random_state=self.seed, algorithm="randomized", n_iter=3).fit(X)
            Z = self.svd.transform(X)
            self.dsc = StandardScaler().fit(np.hstack([Z, self.struct(recs)]))
        else:
            Z = self.svd.transform(X)
        return self.dsc.transform(np.hstack([Z, self.struct(recs)]))


class _Base:
    name = "base"
    eval_only = False

    def __init__(self, view="full", seed=0, featurizer=None):
        self.view, self.seed = view, seed
        self.fz = featurizer or Featurizer(view, seed)

    def logit(self, recs) -> np.ndarray:
        p = np.clip(self.predict_proba(recs), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))


class TfidfLR(_Base):
    name = "tfidf_lr"

    def fit(self, recs, y):
        self.fz.fit(recs)
        self.m = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear")
        self.m.fit(self._X(recs), y)
        return self

    def _X(self, recs):
        return sp.hstack([self.fz.sparse(recs), sp.csr_matrix(self.fz.struct(recs))]).tocsr()

    def predict_proba(self, recs):
        return self.m.predict_proba(self._X(recs))[:, 1]

    def logit(self, recs):
        return self.m.decision_function(self._X(recs))


class SmallMLP(_Base):
    name = "mlp"

    def fit(self, recs, y):
        D = self.fz.fit(recs).dense(recs)
        self.m = MLPClassifier((256, 64), alpha=1e-3, batch_size=256, max_iter=40, early_stopping=True,
                               n_iter_no_change=4, random_state=self.seed)
        self.m.fit(D, y)
        return self

    def predict_proba(self, recs):
        return self.m.predict_proba(self.fz.dense(recs))[:, 1]


class GBM(_Base):
    name = "gbm"

    def fit(self, recs, y):
        D = self.fz.fit(recs).dense(recs)
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                early_stopping=True, random_state=self.seed)
        self.m.fit(D, y)
        return self

    def predict_proba(self, recs):
        return self.m.predict_proba(self.fz.dense(recs))[:, 1]


JUDGES = {"tfidf_lr": TfidfLR, "mlp": SmallMLP, "gbm": GBM}
