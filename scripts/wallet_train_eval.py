"""Train/calibrate/threshold judges on wallet scenarios and report drain success with/without firewall.
Usage: python scripts/wallet_train_eval.py"""
import collections
import gzip
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afw.calibration import brier, budget_thresholds, fit_temperature, reliability, sigmoid  # noqa
from afw.datarules import check_trainable  # noqa
from afw.firewall import Firewall  # noqa
from afw.judges.cpu import JUDGES, Featurizer  # noqa

EXPS = {"random": None, "heldout_attack": {"permit_phish", "direct_transfer", "wrong_verified_router"},
        "heldout_intent": {"claim", "revoke", "explicit_address"}}


def split(r):
    m = r["seed"] % 10
    return "train" if m < 6 else ("val" if m < 8 else "test")


def select(recs, exp, part):
    out = [r for r in recs if split(r) == part]
    held = EXPS[exp]
    if held is None:
        return out
    key = "intent_type" if exp == "heldout_intent" else "attack"
    if part != "test":
        return [r for r in out if r[key] not in held]
    return [r for r in out if r[key] in held or (key == "attack" and r["label"] == 0)]


def rates(v, y):
    v = np.asarray(v)
    pos, neg = y == 1, y == 0
    f = lambda m, s: float((v[m] == s).mean()) if m.sum() else float("nan")  # noqa
    return {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "catch_block": f(pos, "block"),
            "catch_block_or_ask": float(np.isin(v[pos], ["block", "ask"]).mean()) if pos.sum() else float("nan"),
            "false_block": f(neg, "block"), "false_ask": f(neg, "ask")}


def main():
    recs = [json.loads(l) for l in gzip.open("data/wallet_records.jsonl.gz", "rt")]
    check_trainable(recs)
    results = []
    for exp in EXPS:
        tr, va, te = (select(recs, exp, p) for p in ("train", "val", "test"))
        ytr, yva, yte = (np.array([r["label"] for r in s]) for s in (tr, va, te))
        fzs = {}
        for view, model in [("wallet", "tfidf_lr"), ("wallet", "mlp"), ("wallet", "gbm"), ("wallet_raw", "tfidf_lr")]:
            fz = fzs.setdefault(view, Featurizer(view, n_svd=96))
            j = JUDGES[model](view=view, featurizer=fz).fit(tr, ytr)
            lva = j.logit(va)
            T = fit_temperature(lva, yva)
            th = budget_thresholds(sigmoid(lva / T), yva, np.array(["sign"] * len(va)))
            fw = Firewall(j, T, th)
            lte = j.logit(te)
            praw, p = sigmoid(lte), sigmoid(lte / T)
            v = [fw.decide_p(float(x), "sign", r["risk"]) for x, r in zip(p, te)]
            # on-chain outcome: an ask is resolved by a user who denies injected requests
            executed = [vv == "allow" or (vv == "ask" and r["label"] == 0) for vv, r in zip(v, te)]
            mal = [r for r in te if r["label"] == 1]
            ep = {"drain_success_no_fw": float(np.mean([r["attack_success_if_executed"] for r in mal])),
                  "drain_success_fw": float(np.mean([r["attack_success_if_executed"] and ex for r, ex in zip(te, executed) if r["label"] == 1])),
                  "attacker_gain_no_fw": float(sum(r["attacker_gain"] for r in te)),
                  "attacker_gain_fw": float(sum(r["attacker_gain"] for r, ex in zip(te, executed) if ex)),
                  "legit_completed_fw": float(np.mean([ex for r, ex in zip(te, executed) if r["label"] == 0]))}
            bd = {}
            for key in ("attack", "intent_type", "roman_urdu", "unlisted_dapp"):
                g = collections.defaultdict(list)
                for i, r in enumerate(te):
                    g[str(r[key])].append(i)
                bd[key] = {k: rates(np.asarray(v)[idx], yte[idx]) for k, idx in sorted(g.items())}
            rr, er = reliability(praw, yte)
            rc, ec = reliability(p, yte)
            m = {"experiment": exp, "model": model, "view": view, "n_train": len(tr), "n_test": len(te), "temperature": T,
                 "thresholds": th, "auroc": float(roc_auc_score(yte, p)), "auprc": float(average_precision_score(yte, p)),
                 "ece_raw": er, "ece_cal": ec, "brier_cal": brier(p, yte), "reliability_raw": rr, "reliability_cal": rc,
                 "overall": rates(v, yte), "episodes": ep, "breakdown": bd}
            results.append(m)
            print(f"{exp:15s} {model:9s} {view:10s} auroc={m['auroc']:.4f} ece {er:.4f}->{ec:.4f} catch={m['overall']['catch_block_or_ask']:.3f} "
                  f"fb={m['overall']['false_block']:.4f} drain {ep['drain_success_no_fw']:.3f}->{ep['drain_success_fw']:.3f}", flush=True)
    json.dump(results, open("results/wallet_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
