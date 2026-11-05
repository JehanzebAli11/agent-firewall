"""Temperature scaling, reliability statistics and false-alarm-budget thresholds."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(logT):
        p = np.clip(sigmoid(logits / np.exp(logT)), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf, acc = p[m].mean(), y[m].mean()
        ece += m.mean() * abs(conf - acc)
        rows.append((float(conf), float(acc), int(m.sum())))
    return rows, float(ece)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


# "Stricter for pay and sign": same 1% false-BLOCK budget everywhere, but pay/sign spend a wider
# false-ASK budget, i.e. more uncertain high-stakes actions are routed to the user.
BUDGETS = {"general": {"block": 0.01, "ask": 0.02},
           "pay": {"block": 0.01, "ask": 0.05},
           "sign": {"block": 0.01, "ask": 0.05}}


def budget_thresholds(p: np.ndarray, y: np.ndarray, cls: np.ndarray, budgets=BUDGETS) -> dict:
    """Thresholds on calibrated P(misaligned) chosen on held-in validation data so that at most
    `budget` of LEGITIMATE actions (y==0) of each class score strictly above them (verdict uses p > t)."""
    out = {}
    for c, b in budgets.items():
        neg = p[(y == 0) & (cls == c)]
        if len(neg) < 50:  # fall back to pooled negatives
            neg = p[y == 0]
        out[c] = {"block": float(np.quantile(neg, 1 - b["block"], method="higher")),
                  "ask": float(np.quantile(neg, 1 - b["ask"], method="higher"))}
        out[c]["ask"] = min(out[c]["ask"], out[c]["block"])
    return out
