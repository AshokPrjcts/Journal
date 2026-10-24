"""
models.py — Zero-dependency ML models, a knowledge-based ensemble, metrics,
and a decision-intelligence layer. Pure Python standard library only.

Implements:
  * StandardScaler            — feature standardization
  * LogisticRegression        — gradient-descent baseline
  * DecisionTree / RandomForest (bagged shallow trees) — baseline ensemble
  * KnowledgeGuidedGBM        — gradient-boosted stumps initialized/biased by a
                                knowledge prior => the "knowledge-based ML" model
  * metrics (accuracy, precision, recall, F1, ROC curve, AUC)
  * DecisionPolicy            — cost-sensitive threshold optimization
                                (the "decision intelligence" layer)
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple

Vector = List[float]
Matrix = List[List[float]]


# ----------------------------- preprocessing -------------------------------
class StandardScaler:
    def fit(self, X: Matrix):
        n, d = len(X), len(X[0])
        self.mean = [sum(r[j] for r in X) / n for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((r[j] - self.mean[j]) ** 2 for r in X) / n
            self.std.append(math.sqrt(var) or 1.0)
        return self

    def transform(self, X: Matrix) -> Matrix:
        return [[(r[j] - self.mean[j]) / self.std[j] for j in range(len(r))]
                for r in X]

    def fit_transform(self, X: Matrix) -> Matrix:
        return self.fit(X).transform(X)


# ----------------------------- logistic regression -------------------------
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=400, l2=1e-3, seed=0):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed

    def fit(self, X: Matrix, y: List[int]):
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                z = self.b + sum(self.w[j] * X[i][j] for j in range(d))
                p = 1.0 / (1.0 + math.exp(-z))
                err = p - y[i]
                for j in range(d):
                    gw[j] += err * X[i][j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * gb / n
        return self

    def predict_proba(self, X: Matrix) -> Vector:
        out = []
        for r in X:
            z = self.b + sum(self.w[j] * r[j] for j in range(len(r)))
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out


# ----------------------------- decision tree -------------------------------
class _Node:
    __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self):
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None
        self.value = None  # leaf probability


class DecisionTree:
    def __init__(self, max_depth=4, min_leaf=10, n_features=None, seed=0):
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.n_features, self.rng = n_features, random.Random(seed)

    @staticmethod
    def _gini(y):
        if not y:
            return 0.0
        p = sum(y) / len(y)
        return 1.0 - (p * p + (1 - p) * (1 - p))

    def _best_split(self, X, y, feats):
        best = (None, None, 1e9)
        base_n = len(y)
        for f in feats:
            vals = sorted(set(r[f] for r in X))
            for a, b in zip(vals, vals[1:]):
                thr = (a + b) / 2
                ly = [y[i] for i in range(base_n) if X[i][f] <= thr]
                ry = [y[i] for i in range(base_n) if X[i][f] > thr]
                if len(ly) < self.min_leaf or len(ry) < self.min_leaf:
                    continue
                g = (len(ly) * self._gini(ly) + len(ry) * self._gini(ry)) / base_n
                if g < best[2]:
                    best = (f, thr, g)
        return best

    def _build(self, X, y, depth):
        node = _Node()
        node.value = sum(y) / len(y) if y else 0.0
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf \
                or node.value in (0.0, 1.0):
            return node
        d = len(X[0])
        k = self.n_features or d
        feats = self.rng.sample(range(d), min(k, d))
        f, thr, _ = self._best_split(X, y, feats)
        if f is None:
            return node
        li = [i for i in range(len(X)) if X[i][f] <= thr]
        ri = [i for i in range(len(X)) if X[i][f] > thr]
        node.feat, node.thr = f, thr
        node.left = self._build([X[i] for i in li], [y[i] for i in li], depth + 1)
        node.right = self._build([X[i] for i in ri], [y[i] for i in ri], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _p(self, node, r):
        while node.feat is not None:
            node = node.left if r[node.feat] <= node.thr else node.right
        return node.value

    def predict_proba(self, X):
        return [self._p(self.root, r) for r in X]


class RandomForest:
    def __init__(self, n_trees=25, max_depth=5, min_leaf=8, seed=0):
        self.n_trees, self.max_depth, self.min_leaf, self.seed = \
            n_trees, max_depth, min_leaf, seed

    def fit(self, X, y):
        rng = random.Random(self.seed)
        n, d = len(X), len(X[0])
        self.trees = []
        for t in range(self.n_trees):
            ids = [rng.randrange(n) for _ in range(n)]  # bootstrap
            bx, by = [X[i] for i in ids], [y[i] for i in ids]
            tree = DecisionTree(self.max_depth, self.min_leaf,
                                n_features=max(2, int(math.sqrt(d))),
                                seed=self.seed + t).fit(bx, by)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        acc = [0.0] * len(X)
        for tree in self.trees:
            for i, p in enumerate(tree.predict_proba(X)):
                acc[i] += p
        return [a / len(self.trees) for a in acc]


# -------------------- knowledge-guided gradient boosting -------------------
class KnowledgeGuidedGBM:
    """Gradient-boosted decision stumps whose initial score is a *knowledge
    prior* — a theory-informed weighted combination of standardized features.
    This injects domain knowledge (digital-transformation constructs) directly
    into the learner, yielding the paper's "knowledge-based ML" model.
    """

    def __init__(self, n_rounds=60, lr=0.2, max_depth=2, min_leaf=12,
                 subsample=0.7, knowledge_weights=None, feature_names=None,
                 seed=0):
        self.n_rounds, self.lr = n_rounds, lr
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.subsample = subsample  # stochastic gradient boosting (Friedman 2002)
        self.kw, self.fn, self.seed = knowledge_weights, feature_names, seed

    def _knowledge_prior(self, X):
        """Standardized-feature weighted score -> prior log-odds.

        The knowledge prior encodes the theory-informed linear component of
        digital-transformation readiness, giving the ensemble a strong,
        interpretable starting point before boosting the nonlinear residual.
        """
        if not self.kw or not self.fn:
            return [0.0] * len(X)
        w = [self.kw.get(name, 0.0) for name in self.fn]
        # scale knowledge score into a sensible log-odds range
        return [1.3 * sum(w[j] * r[j] for j in range(len(r))) for r in X]

    def fit(self, X, y):
        n = len(X)
        rng = random.Random(self.seed)
        prior = self._knowledge_prior(X)
        self.F0_scale = 1.0
        self.trees = []
        F = [self.F0_scale * prior[i] for i in range(n)]
        m = max(self.min_leaf * 2, int(self.subsample * n))
        for t in range(self.n_rounds):
            # negative gradient of logloss = y - sigmoid(F)
            resid = [y[i] - 1.0 / (1.0 + math.exp(-F[i])) for i in range(n)]
            # stochastic subsample of rows for this round -> variance reduction
            idx = rng.sample(range(n), m)
            sx = [X[i] for i in idx]
            sr = [resid[i] for i in idx]
            tree = _RegStump(self.max_depth, self.min_leaf,
                             seed=self.seed + t).fit(sx, sr)
            upd = tree.predict(X)
            for i in range(n):
                F[i] += self.lr * upd[i]
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        prior = self._knowledge_prior(X)
        F = [self.F0_scale * prior[i] for i in range(len(X))]
        for tree in self.trees:
            upd = tree.predict(X)
            for i in range(len(X)):
                F[i] += self.lr * upd[i]
        return F

    def predict_proba(self, X):
        return [1.0 / (1.0 + math.exp(-f)) for f in self.decision_function(X)]


class _RegStump:
    """Shallow regression tree (for boosting residuals)."""

    def __init__(self, max_depth=2, min_leaf=12, seed=0):
        self.max_depth, self.min_leaf, self.rng = max_depth, min_leaf, random.Random(seed)

    def _var(self, ys):
        if not ys:
            return 0.0
        m = sum(ys) / len(ys)
        return sum((v - m) ** 2 for v in ys)

    def _build(self, X, y, depth):
        node = _Node()
        node.value = sum(y) / len(y) if y else 0.0
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf:
            return node
        best = (None, None, 1e18)
        nrows = len(X)
        # Candidate thresholds: fixed quantile grid per feature (fast, O(1) scan)
        for f in range(len(X[0])):
            col = sorted(r[f] for r in X)
            qs = [col[int(q * (nrows - 1))] for q in
                  (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
            seen = set()
            for thr in qs:
                if thr in seen:
                    continue
                seen.add(thr)
                ly = ry = 0
                lsum = rsum = 0.0
                lsq = rsq = 0.0
                for i in range(nrows):
                    v = y[i]
                    if X[i][f] <= thr:
                        ly += 1; lsum += v; lsq += v * v
                    else:
                        ry += 1; rsum += v; rsq += v * v
                if ly < self.min_leaf or ry < self.min_leaf:
                    continue
                # within-node SSE = sum(sq) - sum^2/n
                sse = (lsq - lsum * lsum / ly) + (rsq - rsum * rsum / ry)
                if sse < best[2]:
                    best = (f, thr, sse)
        f, thr, _ = best
        if f is None:
            return node
        li = [i for i in range(nrows) if X[i][f] <= thr]
        ri = [i for i in range(nrows) if X[i][f] > thr]
        node.feat, node.thr = f, thr
        node.left = self._build([X[i] for i in li], [y[i] for i in li], depth + 1)
        node.right = self._build([X[i] for i in ri], [y[i] for i in ri], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def predict(self, X):
        out = []
        for r in X:
            node = self.root
            while node.feat is not None:
                node = node.left if r[node.feat] <= node.thr else node.right
            out.append(node.value)
        return out


# ------------------------------- metrics -----------------------------------
def confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


def classification_metrics(y_true, proba, thr=0.5):
    y_pred = [1 if p >= thr else 0 for p in proba]
    tp, tn, fp, fn = confusion(y_true, y_pred)
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_curve(y_true, proba):
    """Return (fpr_list, tpr_list) across thresholds."""
    pairs = sorted(zip(proba, y_true), reverse=True)
    P = sum(y_true)
    N = len(y_true) - P
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for p, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P if P else 0.0)
        fpr.append(fp / N if N else 0.0)
    return fpr, tpr


def auc(fpr, tpr):
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return area


# ----------------------- decision-intelligence layer -----------------------
class DecisionPolicy:
    """Cost-sensitive threshold optimization — converts model probabilities
    into managerial decisions by minimizing expected misclassification cost.

    cost_fp: cost of wrongly flagging a low-readiness firm as ready
    cost_fn: cost of missing a genuinely ready firm (opportunity cost)
    In digital-transformation decisions, missing a ready firm (FN) is usually
    costlier than a false alarm, so cost_fn > cost_fp by default.
    """

    def __init__(self, cost_fp=1.0, cost_fn=3.0):
        self.cost_fp, self.cost_fn = cost_fp, cost_fn

    def optimal_threshold(self, y_true, proba, grid=101):
        best_thr, best_cost = 0.5, float("inf")
        curve = []
        for k in range(grid):
            thr = k / (grid - 1)
            m = classification_metrics(y_true, proba, thr)
            cost = self.cost_fp * m["fp"] + self.cost_fn * m["fn"]
            curve.append((thr, cost))
            if cost < best_cost:
                best_cost, best_thr = cost, thr
        return best_thr, best_cost, curve
