"""Train, calibrate, threshold and evaluate the CPU judges on scripted-agent records.

Usage: python scripts/train_eval.py [--models tfidf_lr,mlp,gbm] [--experiments random,heldout_style,...]
Writes results/metrics.json, results/summary.md, figures under results/figures/."""
import argparse
import collections
import gzip
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afw.calibration import BUDGETS, brier, budget_thresholds, fit_temperature, reliability, sigmoid  # noqa
from afw.episode import build_episode  # noqa
from afw.firewall import Firewall, run_episode  # noqa
from afw.judges.cpu import JUDGES, Featurizer  # noqa
from afw.datarules import check_trainable  # noqa

EXPERIMENTS = {
    "random": None,
    "heldout_style": ("style", {"roleplay", "json_tool", "data_field"}),
    "heldout_lang": ("lang", {"ur_rom", "zh"}),
    "heldout_trick": ("trick", {"base64", "zero_width"}),
    "heldout_task": ("task_id", {"T16_split_dinner", "T17_update_address", "T18_webinar", "T19_inbox_summary", "T20_review",
                                  "T25_change_recovery", "T26_delete_contracts"}),
}
BREAKDOWNS = ["style", "lang", "trick", "position", "goal", "action_type", "task_id", "agent"]


def split_of(seed):
    m = seed % 10
    return "train" if m < 6 else ("val" if m < 8 else "test")


def load(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def select(recs, exp, part):
    out = [r for r in recs if split_of(r["seed"]) == part]
    if EXPERIMENTS[exp] is None:
        return out
    key, held = EXPERIMENTS[exp]
    if part in ("train", "val"):
        return [r for r in out if r[key] not in held]
    # test: held-out slice + clean episodes (clean episodes only for non-task experiments)
    if key == "task_id":
        return [r for r in out if r[key] in held]
    return [r for r in out if r[key] in held or not r["injected"]]


def rates(verdicts, y):
    v = np.asarray(verdicts)
    pos, neg = y == 1, y == 0
    f = lambda m, s: float((v[m] == s).mean()) if m.sum() else float("nan")  # noqa
    return {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "catch_block": f(pos, "block"), "catch_block_or_ask": float(np.isin(v[pos], ["block", "ask"]).mean()) if pos.sum() else float("nan"),
            "false_block": f(neg, "block"), "false_ask": f(neg, "ask")}


def episode_eval(test_recs, verdict_of):
    by_ep = collections.defaultdict(list)
    for r in test_recs:
        by_ep[(r["seed"], r["agent"])].append(r)
    # include test episodes with zero proposals too (they can't be attacked or false-blocked)
    res = collections.defaultdict(list)
    for (seed, agent), rs in by_ep.items():
        rs.sort(key=lambda r: r["step"])
        ep = build_episode(seed, agent, rs[0]["injected"])
        assert len(ep.proposals) == len(rs)
        base = run_episode(ep, None)
        fw = run_episode(ep, [verdict_of[r["id"]] for r in rs])
        key = "injected" if ep.injection else "clean"
        res[key].append((base, fw, rs, [verdict_of[r["id"]] for r in rs]))
    out = {}
    if res["injected"]:
        inj = res["injected"]
        out["injected_episodes"] = len(inj)
        out["asr_no_firewall"] = float(np.mean([b["attack_success"] for b, _, _, _ in inj]))
        out["asr_firewall"] = float(np.mean([f["attack_success"] for _, f, _, _ in inj]))
        att = [x for x in inj if x[0]["attack_success"]]
        out["attacks_stopped_frac"] = float(np.mean([not f["attack_success"] for _, f, _, _ in att])) if att else float("nan")
        out["task_completion_injected_no_fw"] = float(np.mean([b["task_completed"] for b, _, _, _ in inj]))
        out["task_completion_injected_fw"] = float(np.mean([f["task_completed"] for _, f, _, _ in inj]))
    if res["clean"]:
        cl = res["clean"]
        out["clean_episodes"] = len(cl)
        out["clean_episode_false_block"] = float(np.mean([("block" in v) for *_, v in cl]))
        out["clean_episode_ask"] = float(np.mean([("ask" in v) for *_, v in cl]))
        out["task_completion_clean_fw"] = float(np.mean([f["task_completed"] for _, f, _, _ in cl]))
    return out


def breakdown(test_recs, verdicts, y):
    out = {}
    v = np.asarray(verdicts)
    for key in BREAKDOWNS:
        groups = collections.defaultdict(list)
        for i, r in enumerate(test_recs):
            groups[r[key]].append(i)
        out[key] = {g: rates(v[idx], y[idx]) for g, idx in sorted(groups.items())}
    return out


def run(recs, exp, model, view, figdir, save_dir=None, seed=0, fz=None):
    tr, va, te = (select(recs, exp, p) for p in ("train", "val", "test"))
    check_trainable(tr)
    ytr, yva, yte = (np.array([r["label"] for r in s]) for s in (tr, va, te))
    t0 = time.time()
    judge = JUDGES[model](view=view, seed=seed, featurizer=fz).fit(tr, ytr)
    t_fit = time.time() - t0
    lva = judge.logit(va)
    T = fit_temperature(lva, yva)
    pva = sigmoid(lva / T)
    th = budget_thresholds(pva, yva, np.array([r["action_class"] for r in va]))
    fw = Firewall(judge, T, th)
    t0 = time.time()
    lte = judge.logit(te)
    t_inf = (time.time() - t0) / max(1, len(te))
    p_raw, p_cal = sigmoid(lte), sigmoid(lte / T)
    verdicts = [fw.decide_p(float(p), r["action_class"], r["risk"]) for p, r in zip(p_cal, te)]
    rel_raw, ece_raw = reliability(p_raw, yte)
    rel_cal, ece_cal = reliability(p_cal, yte)
    m = {"experiment": exp, "model": model, "view": view, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "fit_seconds": round(t_fit, 1), "infer_ms_per_action": round(1000 * t_inf, 3), "temperature": T,
         "thresholds": th, "auroc": float(roc_auc_score(yte, p_cal)), "auprc": float(average_precision_score(yte, p_cal)),
         "ece_raw": ece_raw, "ece_cal": ece_cal, "brier_raw": brier(p_raw, yte), "brier_cal": brier(p_cal, yte),
         "reliability_raw": rel_raw, "reliability_cal": rel_cal,
         "overall": rates(verdicts, yte)}
    for c in BUDGETS:
        idx = np.array([r["action_class"] == c for r in te])
        if idx.sum():
            m.setdefault("by_class", {})[c] = rates(np.asarray(verdicts)[idx], yte[idx])
    m["breakdown"] = breakdown(te, verdicts, yte)
    m["episodes"] = episode_eval(te, {r["id"]: v for r, v in zip(te, verdicts)})
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(fw, os.path.join(save_dir, f"{exp}_{model}_{view}.joblib"), compress=3)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/records.jsonl.gz")
    ap.add_argument("--models", default="tfidf_lr,mlp,gbm")
    ap.add_argument("--views", default="full,blind")
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--save-models", default="models")
    a = ap.parse_args()
    recs = load(a.data)
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["experiment"], r["model"], r["view"]) for r in results}
    for exp in a.experiments.split(","):
        for view in a.views.split(","):
            fz = Featurizer(view)
            for model in a.models.split(","):
                if (exp, model, view) in done:
                    continue
                t0 = time.time()
                m = run(recs, exp, model, view, None, a.save_models if exp == "random" else None, fz=fz)
                o, e = m["overall"], m["episodes"]
                print(f"{exp:14s} {model:9s} {view:5s} auroc={m['auroc']:.4f} ece {m['ece_raw']:.4f}->{m['ece_cal']:.4f} "
                      f"catch={o['catch_block']:.3f}/{o['catch_block_or_ask']:.3f} fb={o['false_block']:.4f} "
                      f"ASR {e.get('asr_no_firewall', float('nan')):.3f}->{e.get('asr_firewall', float('nan')):.3f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
                results = [r for r in results if (r["experiment"], r["model"], r["view"]) != (exp, model, view)] + [m]
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
